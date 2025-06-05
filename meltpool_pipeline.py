import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18
from tqdm.auto import tqdm


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


def train_model(csv_path: str, img_dir: str, transform: T.Compose, tag: str, device: str) -> None:
    full_ds = MeltpoolDataset(csv_path=csv_path, img_dir=img_dir, transform=transform)
    val_frac = 0.20
    n_val = int(len(full_ds) * val_frac)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val],
                                                    generator=torch.Generator().manual_seed(42))
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


def main(csv_path: str, img_dir: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for tag, tfm in mask_transforms.items():
        print(f'=== TRAINING START: {tag} ===')
        train_model(csv_path, img_dir, tfm, tag, device)
    print('=== ALL TRAINING DONE ===')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train meltpool distance models with various masks')
    parser.add_argument('--csv', required=True, help='Path to labels_train.csv')
    parser.add_argument('--imgs', required=True, help='Path to directory with TIFF images')
    args = parser.parse_args()

    main(args.csv, args.imgs)
