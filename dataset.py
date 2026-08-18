"""Veri yukleme, train/val/test bolme ve veri artirma.

Severstal train.csv iki farkli formatta dagitildi:
  (A) ImageId_ClassId, EncodedPixels     (her goruntu icin 4 satir)
  (B) ImageId, ClassId, EncodedPixels    (sadece kusurlu satirlar)
Asagidaki kod her ikisini de destekler.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import rle_to_mask

cv2.setNumThreads(0)  # DataLoader worker'lari ile cakismasin


# --------------------------------------------------------------------------
# 1) Goruntu basina 4 sutunlu bir tablo olustur
# --------------------------------------------------------------------------
def build_dataframe(data_dir, num_classes=4):
    csv_path = os.path.join(data_dir, "train.csv")
    img_dir = os.path.join(data_dir, "train_images")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"train.csv bulunamadi: {csv_path}\n"
            "--data_dir parametresini kontrol et. Kaggle'da dogru yol:\n"
            "/kaggle/input/severstal-steel-defect-detection"
        )

    raw = pd.read_csv(csv_path)
    if "ImageId_ClassId" in raw.columns:                       # format (A)
        raw["ImageId"] = raw["ImageId_ClassId"].str.rsplit("_", n=1).str[0]
        raw["ClassId"] = raw["ImageId_ClassId"].str.rsplit("_", n=1).str[1].astype(int)
    raw = raw[raw["EncodedPixels"].notna()]
    raw["ClassId"] = raw["ClassId"].astype(int)

    # Klasordeki TUM goruntuler (kusursuzlar csv'de olmayabilir)
    all_images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    df = pd.DataFrame({"ImageId": all_images})

    for c in range(1, num_classes + 1):
        sub = raw[raw["ClassId"] == c].set_index("ImageId")["EncodedPixels"]
        df[f"rle{c}"] = df["ImageId"].map(sub)

    # Etiket vektoru: hangi siniflar var?
    for c in range(1, num_classes + 1):
        df[f"has{c}"] = df[f"rle{c}"].notna().astype(int)
    df["n_defects"] = df[[f"has{c}" for c in range(1, num_classes + 1)]].sum(axis=1)
    # Stratified split icin imza: "0100" gibi
    df["signature"] = df[[f"has{c}" for c in range(1, num_classes + 1)]].astype(str).agg("".join, axis=1)
    return df


# --------------------------------------------------------------------------
# 2) Stratified train / val / test bolme
# --------------------------------------------------------------------------
def split_dataframe(df, val_size=0.15, test_size=0.15, seed=42, min_per_group=20):
    """Stratified 70/15/15 bolme.

    Severstal'da bazi kusur kombinasyonlari (orn. class1+class2 birlikte) cok nadir.
    Bir grupta 3'ten az ornek varsa stratified split matematiksel olarak imkansiz
    (her bolmeye en az 1 ornek dusmeli). Bu yuzden:
      1. Nadir imzalari tek bir 'rare' kovasinda topluyoruz
      2. Yine de yetmezse kusur SAYISINA gore stratify ediyoruz (daha kaba ama saglam)
      3. O da olmazsa stratify olmadan rastgele boluyoruz
    """
    from sklearn.model_selection import train_test_split

    def _make_strata(threshold):
        counts = df["signature"].value_counts()
        rare = counts[counts < threshold].index
        s = df["signature"].where(~df["signature"].isin(rare), "rare")
        # 'rare' kovasi da 3'ten kucukse stratify yine patlar -> en yaygin imzaya kat
        vc = s.value_counts()
        tiny = vc[vc < 3].index
        if len(tiny):
            s = s.where(~s.isin(tiny), vc.idxmax())
        return s

    candidates = [
        ("signature", _make_strata(min_per_group)),
        ("n_defects", df["n_defects"].astype(str)),
        ("none", None),
    ]

    for name, strat in candidates:
        try:
            if strat is None:
                train_df, temp_df = train_test_split(
                    df, test_size=val_size + test_size, random_state=seed)
                rel = test_size / (val_size + test_size)
                val_df, test_df = train_test_split(temp_df, test_size=rel, random_state=seed)
            else:
                train_df, temp_df, _, strat_tmp = train_test_split(
                    df, strat, test_size=val_size + test_size,
                    random_state=seed, stratify=strat)
                rel = test_size / (val_size + test_size)
                val_df, test_df = train_test_split(
                    temp_df, test_size=rel, random_state=seed, stratify=strat_tmp)
            if name != "signature":
                print(f"[bilgi] stratify stratejisi: '{name}' (imza bazli bolme mumkun olmadi)")
            return (train_df.reset_index(drop=True),
                    val_df.reset_index(drop=True),
                    test_df.reset_index(drop=True))
        except ValueError as e:
            print(f"[bilgi] '{name}' ile stratify basarisiz ({e}); bir sonraki stratejiye geciliyor.")
            continue

    raise RuntimeError("Veri bolunemedi.")


# --------------------------------------------------------------------------
# 3) Veri artirma (regularization yontemlerinden biri)
# --------------------------------------------------------------------------
def get_transforms(train, cfg):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # ImageNet istatistikleri (encoder on egitimli oldugu icin ayni normalizasyon)
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    if train and int(cfg.augment):
        return A.Compose([
            A.RandomCrop(height=cfg.img_h, width=cfg.crop_w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            norm, ToTensorV2(),
        ])
    if train:  # augmentasyon kapali ablasyonu: sadece kirpma (hiz icin) + normalize
        return A.Compose([A.RandomCrop(height=cfg.img_h, width=cfg.crop_w), norm, ToTensorV2()])
    # Dogrulama/test: tam cozunurluk, hicbir rastgelelik yok
    return A.Compose([norm, ToTensorV2()])


# --------------------------------------------------------------------------
# 4) Dataset
# --------------------------------------------------------------------------
class SteelDataset(Dataset):
    def __init__(self, df, data_dir, cfg, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = os.path.join(data_dir, "train_images")
        self.cfg = cfg
        self.tf = get_transforms(train, cfg)
        self.nc = cfg.num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["ImageId"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros((self.cfg.img_h, self.cfg.img_w, self.nc), dtype=np.uint8)
        for c in range(self.nc):
            mask[:, :, c] = rle_to_mask(row[f"rle{c+1}"], self.cfg.img_h, self.cfg.img_w)

        out = self.tf(image=img, mask=mask)
        image = out["image"].float()
        mask = out["mask"].permute(2, 0, 1).float()          # (H,W,C) -> (C,H,W)
        label = (mask.sum(dim=(1, 2)) > 0).float()           # kirpma sonrasi sinif etiketi
        return image, mask, label


def make_loaders(cfg):
    from torch.utils.data import DataLoader

    df = build_dataframe(cfg.data_dir, cfg.num_classes)
    if cfg.limit and cfg.limit > 0:                # hizli smoke test
        df = df.sample(n=min(cfg.limit, len(df)), random_state=cfg.seed).reset_index(drop=True)

    train_df, val_df, test_df = split_dataframe(df, cfg.val_size, cfg.test_size, cfg.seed)

    tl = DataLoader(SteelDataset(train_df, cfg.data_dir, cfg, train=True),
                    batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(SteelDataset(val_df, cfg.data_dir, cfg, train=False),
                    batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True)
    sl = DataLoader(SteelDataset(test_df, cfg.data_dir, cfg, train=False),
                    batch_size=cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=True)
    return tl, vl, sl, (train_df, val_df, test_df)
