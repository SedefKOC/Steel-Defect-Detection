"""Yardimci fonksiyonlar: seed sabitleme, RLE kodlama/cozme, grafik cizimi."""
import os
import random
import numpy as np


# --------------------------------------------------------------------------
# Tekrarlanabilirlik (odev "fix random seeds" istiyor)
# --------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# RLE (Run-Length Encoding)
# Kaggle maskeleri "baslangic uzunluk baslangic uzunluk ..." seklinde saklar.
# ONEMLI: indeksleme 1'den baslar ve SUTUN bazlidir (Fortran order).
# --------------------------------------------------------------------------
def rle_to_mask(rle, height=256, width=1600):
    """RLE string -> (H, W) uint8 ikili maske."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if not isinstance(rle, str) or len(rle.strip()) == 0:
        return mask.reshape(height, width, order="F")
    nums = rle.split()
    starts = np.asarray(nums[0::2], dtype=np.int64) - 1  # 1-indeksli -> 0-indeksli
    lengths = np.asarray(nums[1::2], dtype=np.int64)
    for s, l in zip(starts, lengths):
        mask[s:s + l] = 1
    return mask.reshape(height, width, order="F")


def mask_to_rle(mask):
    """(H, W) ikili maske -> RLE string. rle_to_mask'in tersi."""
    pixels = mask.T.flatten()  # sutun bazli okuma
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


# --------------------------------------------------------------------------
# Egitim gecmisi grafikleri (rapordaki overfitting analizi icin sart)
# --------------------------------------------------------------------------
def plot_history(history_csv, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].plot(df["epoch"], df["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(df["epoch"], df["val_loss"], label="validation", marker="o", ms=3)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")

    axes[1].plot(df["epoch"], df["train_dice"], label="train", marker="o", ms=3)
    axes[1].plot(df["epoch"], df["val_dice"], label="validation", marker="o", ms=3)
    axes[1].set_title("Mean Dice vs Epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("dice")

    axes[2].plot(df["epoch"], df["lr"], marker="o", ms=3, color="tab:green")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("lr")
    axes[2].set_yscale("log")

    for ax in axes[:2]:
        ax.legend(); ax.grid(alpha=0.3)
    axes[2].grid(alpha=0.3)

    # En iyi epoch'u isaretle (early stopping'in nerede durdugunu gosterir)
    best = int(df["val_dice"].idxmax())
    for ax in axes[:2]:
        ax.axvline(df["epoch"].iloc[best], ls="--", c="red", alpha=0.6)
    axes[1].annotate(f"best epoch {int(df['epoch'].iloc[best])}\nval dice={df['val_dice'].iloc[best]:.4f}",
                     xy=(df["epoch"].iloc[best], df["val_dice"].iloc[best]),
                     xytext=(0.45, 0.15), textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()
    return out_png


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
