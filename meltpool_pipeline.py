import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from torchvision.models import resnet18
from tqdm.auto import tqdm


def mount_google_drive() -> None:
    """Mount Google Drive inside Colab if possible."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as exc:  # pragma: no cover - colab only
        print(f'Could not mount Google Drive: {exc}')


class MeltpoolDataset(Dataset):
    """Dataset for meltpool distance prediction."""

    def __init__(self, csv_path: str, img_dir: str, transform: T.Compose | None = None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(row['nozzel_distance'], dtype=torch.float32)
        return img, target


def validate_dataset(csv_path: str, img_dir: str) -> pd.DataFrame:
    """Verify that every image listed in the CSV exists in ``img_dir``."""
    df = pd.read_csv(csv_path)
    missing: list[str] = []
    for img_name in df['image']:
        if not os.path.isfile(os.path.join(img_dir, img_name)):
            missing.append(img_name)
    if missing:
        raise FileNotFoundError(f"Missing images: {missing}")
    print(f"Dataset check passed: {len(df)} entries")
    return df


# ----------------------------- mask transforms -----------------------------
class NozzleMask:
    def __init__(self, y_c: int = 250, x_c: int = 320, r: int = 150):
        from skimage.morphology import disk
        self.yc, self.xc, self.r = y_c, x_c, r
        self.disk_mask = disk(r)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        mask = np.zeros_like(arr, dtype=bool)
        y1, y2 = self.yc - self.r, self.yc + self.r + 1
        x1, x2 = self.xc - self.r, self.xc + self.r + 1
        dy1 = max(0, -y1)
        dx1 = max(0, -x1)
        dy2 = min(arr.shape[0] - y1, 2 * self.r + 1)
        dx2 = min(arr.shape[1] - x1, 2 * self.r + 1)
        mask_chunk = self.disk_mask[dy1:dy2, dx1:dx2]
        mask[max(0, y1):min(arr.shape[0], y2), max(0, x1):min(arr.shape[1], x2)] = mask_chunk
        arr[mask] = 0
        return Image.fromarray(arr)


class ThresholdMask:
    def __init__(self, thresh: int = 128):
        self.thresh = thresh

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        bin_arr = (arr > self.thresh).astype(np.uint8) * 255
        return Image.fromarray(bin_arr)


class DBSCANMask:
    def __init__(self, percent: float = 0.1, eps: int = 2, min_samples: int = 10):
        self.percent = percent
        self.eps = eps
        self.min_samples = min_samples

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        flat = arr.flatten()
        thr = np.percentile(flat, 100 * (1 - self.percent))
        ys, xs = np.where(arr >= thr)
        if len(ys) == 0:
            return pil_img
        pts = np.column_stack([ys, xs])
        cl = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts)
        labels = cl.labels_
        counts = np.bincount(labels[labels >= 0])
        if len(counts) == 0:
            return pil_img
        good_label = np.argmax(counts)
        mask = np.zeros_like(arr, dtype=bool)
        mask[pts[labels == good_label, 0], pts[labels == good_label, 1]] = True
        out = np.zeros_like(arr)
        out[mask] = arr[mask]
        return Image.fromarray(out)


class BBoxMask:
    def __init__(self, seg_func):
        self.seg_func = seg_func

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        seg_mask = np.array(self.seg_func(pil_img).convert('L')) > 0
        if not seg_mask.any():
            return pil_img.resize((128, 128))
        ys, xs = np.where(seg_mask)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        crop = arr[y1:y2 + 1, x1:x2 + 1]
        return Image.fromarray(crop).resize((128, 128))


mask_transforms = {
    'baseline': T.Compose([T.Resize((128, 128)), T.ToTensor()]),
    'baseline_aug': T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ToTensor(),
    ]),
    'nozzle': T.Compose([T.Resize((128, 128)), NozzleMask(), T.ToTensor()]),
    'threshold': T.Compose([T.Resize((128, 128)), ThresholdMask(), T.ToTensor()]),
    'dbscan': T.Compose([T.Resize((128, 128)), DBSCANMask(), T.ToTensor()]),
    'bbox': T.Compose([T.Resize((128, 128)), BBoxMask(seg_func=ThresholdMask()), T.ToTensor()]),
    'all_masks': T.Compose([
        T.Resize((128, 128)),
        NozzleMask(),
        ThresholdMask(),
        DBSCANMask(),
        BBoxMask(seg_func=ThresholdMask()),
        T.ToTensor(),
    ]),
    'all_masks_aug': T.Compose([
        T.Resize((128, 128)),
        NozzleMask(),
        ThresholdMask(),
        DBSCANMask(),
        BBoxMask(seg_func=ThresholdMask()),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ToTensor(),
    ])
}


# ----------------------------- training utils -----------------------------
def get_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 1)
    return model


def train_on_dataset(dataset: torch.utils.data.Dataset, tag: str, device: str) -> tuple[int, float]:
    """Train ``resnet18`` on the provided dataset and return best epoch and loss."""
    val_frac = 0.20
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = get_model().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(1, 6):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in tqdm(train_loader, desc=f'{tag} Epoch {epoch} [Train]'):
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                preds = model(xb)
                val_loss_sum += loss_fn(preds, yb).item() * xb.size(0)
        val_loss = val_loss_sum / len(val_loader.dataset)
        print(f'{tag} Epoch {epoch}/5  train_MSE={train_loss:.4f}  val_MSE={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'resnet18_{tag}.pt')
    print(f'{tag} ▶ Best Epoch: {best_epoch}, val_MSE={best_val_loss:.4f}\n')
    return best_epoch, best_val_loss


def train_model(csv_path: str, img_dir: str, transform: T.Compose, tag: str, device: str) -> tuple[int, float]:
    dataset = MeltpoolDataset(csv_path=csv_path, img_dir=img_dir, transform=transform)
    return train_on_dataset(dataset, tag, device)


def train_masks(csv_path: str, img_dir: str, masks: list[str], device: str) -> pd.DataFrame:
    """Train one model per mask and return a summary ``DataFrame``."""
    records: list[dict[str, float | str | int]] = []
    for mask_name in masks:
        tfm = mask_transforms[mask_name]
        print(f'=== TRAINING START: {mask_name} ===')
        best_epoch, best_loss = train_model(csv_path, img_dir, tfm, mask_name, device)
        records.append({'mask': mask_name, 'best_epoch': best_epoch, 'best_val_MSE': best_loss,
                        'weights': f'resnet18_{mask_name}.pt'})
    return pd.DataFrame(records)


def train_combined(csv_path: str, img_dir: str, masks: list[str], device: str) -> tuple[int, float]:
    """Train a single model on a concatenation of the datasets for ``masks``."""
    datasets = [MeltpoolDataset(csv_path, img_dir, mask_transforms[m]) for m in masks]
    combined_ds = ConcatDataset(datasets)
    return train_on_dataset(combined_ds, 'combined', device)


def train_all(csv_path: str, img_dir: str, device: str) -> pd.DataFrame:
    """Backward compatible helper calling :func:`train_masks` for all masks."""
    return train_masks(csv_path, img_dir, list(mask_transforms.keys()), device)



def run_inference_on_directory(img_dir: str, weights: str, transform: T.Compose, device: str) -> pd.DataFrame:
    """Run inference on every image in ``img_dir`` and return predictions."""
    model = get_model().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    records: list[dict[str, float | str]] = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        img = Image.open(os.path.join(img_dir, name)).convert('L')
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).item()
        records.append({'image': name, 'prediction': pred})
        print(f'{name}: {pred:.4f}')
    df = pd.DataFrame(records)
    df.to_csv('drive_predictions.csv', index=False)
    return df


def plot_results(df: pd.DataFrame) -> None:
    """Visualize validation losses for each mask."""
    ax = df.plot(kind='bar', x='mask', y='best_val_MSE', legend=False)
    ax.set_ylabel('Best Validation MSE')
    ax.set_title('Mask comparison')
    plt.tight_layout()
    plt.show()


def main(csv_path: str, img_dir: str, masks: list[str], combine: bool, testdrive: bool, mount_drive: bool) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mount_drive or testdrive:
        mount_google_drive()

    validate_dataset(csv_path, img_dir)
    df = train_masks(csv_path, img_dir, masks, device)

    if combine:
        best_epoch, best_loss = train_combined(csv_path, img_dir, masks, device)
        df = pd.concat([df, pd.DataFrame([{'mask': 'combined', 'best_epoch': best_epoch,
                                           'best_val_MSE': best_loss, 'weights': 'resnet18_combined.pt'}])],
                       ignore_index=True)

    print('=== TRAINING SUMMARY ===')
    print(df.to_string(index=False))

    best_row = df.loc[df['best_val_MSE'].idxmin()]
    print(f"Best model: {best_row['mask']} -> {best_row['best_val_MSE']:.4f}")

    if testdrive:
        drive_dir = '/content/drive/MyDrive/Testdaten/'
        print(f'=== INFERENCE ON DRIVE DATA ({drive_dir}) ===')
        run_inference_on_directory(drive_dir, best_row['weights'],
                                   mask_transforms[best_row['mask']], device)
    plot_results(df)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train meltpool distance models with various masks')
    parser.add_argument('--csv', required=True, help='Path to labels_train.csv')
    parser.add_argument('--imgs', required=True, help='Path to directory with TIFF images')
    parser.add_argument('--masks', nargs='+', default=['baseline', 'nozzle', 'threshold', 'dbscan'],
                        help='Mask names to train individually')
    parser.add_argument('--combine', action='store_true', help='Also train on all masks combined')
    parser.add_argument('--testdrive', action='store_true', help='Run inference on Google Drive test data')
    parser.add_argument('--mount-drive', action='store_true', help='Mount Google Drive (for Colab)')
    args = parser.parse_args()

    main(args.csv, args.imgs, masks=args.masks, combine=args.combine,
         testdrive=args.testdrive, mount_drive=args.mount_drive)
