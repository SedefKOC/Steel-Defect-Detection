"""Degerlendirme: esik/min_size ayari (validation uzerinde), test metrikleri,
karisiklik matrisi, ornek tahmin gorselleri ve rapora hazir markdown tablosu.

Bellek notu: tum tahmin maskelerini saklamak yerine (tam veride ~12 GB olurdu)
her (goruntu, sinif, esik) icin sadece uc sayi tutuyoruz:
    inter = |tahmin n gercek|,  psum = |tahmin|,  tsum = |gercek|
Dice ve IoU bu uc sayidan hesaplanabilir. min_size filtresi de psum'a bakarak
uygulanabilir (alani esigin altindaki tahmin silinir -> inter=0, psum=0).
Boylece tek gecisle tum (esik x min_size) izgarasini tarayabiliyoruz.

Kullanim:
    python src/evaluate.py --run_name final
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, add_config_args, config_from_args
from dataset import SteelDataset, build_dataframe
from model import build_model
from utils import set_seed

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SIZES = [0, 300, 600, 1200, 2000]


# --------------------------------------------------------------------------
@torch.no_grad()
def accumulate_stats(model, loader, device, thresholds, tta=True, amp=True):
    """Tek gecis; her esik icin (N, C) boyutunda inter/psum dizileri toplar."""
    model.eval()
    inter = {t: [] for t in thresholds}
    psum = {t: [] for t in thresholds}
    tsum = []

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(images)
            probs = torch.sigmoid(logits.float())
            if tta:  # Test-Time Augmentation: yatay + dikey cevirme ortalamasi
                for dims in ([3], [2]):
                    lg, _ = model(torch.flip(images, dims=dims))
                    probs = probs + torch.sigmoid(torch.flip(lg.float(), dims=dims))
                probs = probs / 3.0

        tsum.append(masks.sum(dim=(2, 3)).cpu().numpy())
        for t in thresholds:
            pred = (probs > t).float()
            inter[t].append((pred * masks).sum(dim=(2, 3)).cpu().numpy())
            psum[t].append(pred.sum(dim=(2, 3)).cpu().numpy())

    return ({t: np.concatenate(v, 0) for t, v in inter.items()},
            {t: np.concatenate(v, 0) for t, v in psum.items()},
            np.concatenate(tsum, 0))


def dice_from_stats(inter, psum, tsum, min_size):
    """min_size filtresini uygula ve (N, C) dice matrisini dondur."""
    keep = psum >= min_size          # alani kucuk tahminler silinir
    i = np.where(keep, inter, 0.0)
    p = np.where(keep, psum, 0.0)
    denom = p + tsum
    return np.where(denom == 0, 1.0, 2.0 * i / np.maximum(denom, 1e-7)), i, p


def iou_from_stats(i, p, tsum):
    union = p + tsum - i
    return np.where(union == 0, 1.0, i / np.maximum(union, 1e-7))


def evaluate_grid(inter, psum, tsum, thresholds, min_sizes):
    rows, best = [], None
    for t in thresholds:
        for ms in min_sizes:
            d, _, _ = dice_from_stats(inter[t], psum[t], tsum, ms)
            score = float(d.mean())
            rows.append({"threshold": t, "min_size": ms, "val_dice": score})
            if best is None or score > best[0]:
                best = (score, t, ms)
    return best, pd.DataFrame(rows)


def compute_metrics(inter, psum, tsum, threshold, min_size, num_classes=4):
    d, i, p = dice_from_stats(inter[threshold], psum[threshold], tsum, min_size)
    iou = iou_from_stats(i, p, tsum)
    y_true = (tsum > 0).astype(int)
    y_pred = (p > 0).astype(int)

    res = {"mean_dice": float(d.mean()), "mean_iou": float(iou.mean())}
    for c in range(num_classes):
        res[f"dice_class{c+1}"] = float(d[:, c].mean())
        pos = y_true[:, c] == 1
        res[f"dice_pos_class{c+1}"] = float(d[pos, c].mean()) if pos.sum() else float("nan")
    try:
        from sklearn.metrics import precision_recall_fscore_support
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
        for c in range(num_classes):
            res[f"precision_class{c+1}"] = float(pr[c])
            res[f"recall_class{c+1}"] = float(rc[c])
            res[f"f1_class{c+1}"] = float(f1[c])
        res["macro_f1"] = float(np.mean(f1))
    except Exception:
        pass
    return res, y_true, y_pred


# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, out_png, num_classes=4):
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 3.6))
    for c in range(num_classes):
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])
        ax = axes[c]
        ax.imshow(cm, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, cm[a, b], ha="center", va="center",
                        color="white" if cm[a, b] > cm.max() / 2 else "black")
        ax.set_title(f"Class {c+1}")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


@torch.no_grad()
def plot_examples(model, dataset, device, out_png, threshold, min_size, n=4):
    """Sadece birkac ornek icin modeli tekrar calistirip gorsellestirir."""
    import cv2
    model.eval()
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 128, 255], [255, 255, 0]])

    # Kusur iceren ilk n ornegi sec
    idxs = [i for i in range(len(dataset))
            if dataset.df.iloc[i]["n_defects"] > 0][:n] or list(range(min(n, len(dataset))))
    if not idxs:
        return

    fig, axes = plt.subplots(len(idxs), 1, figsize=(16, 2.6 * len(idxs)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idxs):
        image, mask, _ = dataset[i]
        logits, _ = model(image.unsqueeze(0).to(device))
        pr = (torch.sigmoid(logits.float())[0] > threshold).cpu().numpy().astype(np.float32)
        for c in range(pr.shape[0]):
            if pr[c].sum() < min_size:
                pr[c] = 0
        gt = mask.numpy()

        row = dataset.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(dataset.img_dir, row["ImageId"])),
                           cv2.COLOR_BGR2RGB)
        vis = img.copy()
        for c in range(gt.shape[0]):
            vis[gt[c] > 0] = (0.45 * vis[gt[c] > 0] + 0.55 * colors[c]).astype(np.uint8)
            extra = pr[c] - np.minimum(pr[c], gt[c])   # fazladan tahmin (yanlis pozitif)
            vis[extra > 0] = (0.35 * vis[extra > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
        ax.imshow(vis); ax.axis("off")
        ax.set_title(f"{row['ImageId']} | renkli = gercek maske, macenta = fazladan tahmin",
                     fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--tta", type=int, default=1)
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, cfg.run_name)
    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"),
                      map_location=device, weights_only=False)

    saved = Config(**ckpt["cfg"])
    saved.data_dir, saved.out_dir, saved.run_name = cfg.data_dir, cfg.out_dir, cfg.run_name
    cfg = saved

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Model yuklendi: epoch {ckpt['epoch']}, val dice {ckpt['val_dice']:.4f}")

    df = build_dataframe(cfg.data_dir, cfg.num_classes)
    from torch.utils.data import DataLoader
    datasets, loaders = {}, {}
    for split in ["val", "test"]:
        ids = set(pd.read_csv(os.path.join(out_dir, f"split_{split}.csv"))["ImageId"])
        sub = df[df["ImageId"].isin(ids)].reset_index(drop=True)
        datasets[split] = SteelDataset(sub, cfg.data_dir, cfg, train=False)
        loaders[split] = DataLoader(datasets[split], batch_size=cfg.val_batch_size,
                                    shuffle=False, num_workers=cfg.num_workers)
        print(f"{split}: {len(sub)} goruntu")

    # ---------- 1) Validation uzerinde son islem ayari ----------
    print("Validation taraniyor...")
    vi, vp, vt = accumulate_stats(model, loaders["val"], device, THRESHOLDS, bool(args.tta))
    (best_dice, best_th, best_ms), grid = evaluate_grid(vi, vp, vt, THRESHOLDS, MIN_SIZES)
    grid.to_csv(os.path.join(out_dir, "postprocess_grid.csv"), index=False)
    print(f"En iyi son islem -> threshold={best_th}, min_size={best_ms}, val dice={best_dice:.4f}")

    # ---------- 2) Test metrikleri ----------
    print("Test degerlendiriliyor...")
    ti, tp, tt = accumulate_stats(model, loaders["test"], device, THRESHOLDS, bool(args.tta))
    metrics, y_true, y_pred = compute_metrics(ti, tp, tt, best_th, best_ms, cfg.num_classes)
    base_metrics, _, _ = compute_metrics(ti, tp, tt, 0.5, 0, cfg.num_classes)

    # ---------- 3) Trivial baseline: "hep bos tahmin et" ----------
    empty_dice = float(np.where(tt == 0, 1.0, 0.0).mean())

    metrics.update({"threshold": best_th, "min_size": best_ms, "tta": bool(args.tta),
                    "dice_default_th0.5_ms0": base_metrics["mean_dice"],
                    "baseline_all_empty_dice": empty_dice,
                    "best_val_dice": float(ckpt["val_dice"]), "best_epoch": int(ckpt["epoch"])})
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------- 4) Gorseller ----------
    plot_confusion(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"), cfg.num_classes)
    plot_examples(model, datasets["test"], device,
                  os.path.join(out_dir, "predictions.png"), best_th, best_ms)

    # ---------- 5) Rapora hazir markdown tablo ----------
    lines = [f"## Test sonuclari - run `{cfg.run_name}`", "",
             "| Metrik | Deger |", "|---|---|",
             f"| Mean Dice (tuned th={best_th}, min_size={best_ms}) | **{metrics['mean_dice']:.4f}** |",
             f"| Mean Dice (varsayilan th=0.5, min_size=0) | {base_metrics['mean_dice']:.4f} |",
             f"| Mean IoU | {metrics['mean_iou']:.4f} |",
             f"| Macro F1 (kusur tespiti) | {metrics.get('macro_f1', float('nan')):.4f} |",
             f"| Baseline: hep bos tahmin | {empty_dice:.4f} |", "",
             "### Sinif bazli sonuclar", "",
             "| Sinif | Dice (tum) | Dice (sadece kusurlu) | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|"]
    for c in range(1, cfg.num_classes + 1):
        lines.append(
            f"| Class {c} | {metrics[f'dice_class{c}']:.4f} | {metrics[f'dice_pos_class{c}']:.4f} | "
            f"{metrics.get(f'precision_class{c}', float('nan')):.4f} | "
            f"{metrics.get(f'recall_class{c}', float('nan')):.4f} | "
            f"{metrics.get(f'f1_class{c}', float('nan')):.4f} |")
    with open(os.path.join(out_dir, "results_table.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nTum ciktilar: {out_dir}")


if __name__ == "__main__":
    main()
