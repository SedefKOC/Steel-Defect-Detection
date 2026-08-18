"""Degerlendirme: esik/min_size ayari (validation uzerinde), test metrikleri,
karisiklik matrisi, ornek tahmin gorselleri ve rapora hazir markdown tablosu.

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
from metrics import MetricTracker, dice_per_sample
from model import build_model
from utils import set_seed


@torch.no_grad()
def collect_predictions(model, loader, device, amp=True, tta=False):
    """Tum tahminleri (olasilik) ve gercek maskeleri CPU'da topla."""
    model.eval()
    probs_all, masks_all = [], []
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(images)
            p = torch.sigmoid(logits.float())
            if tta:  # Test-Time Augmentation: yatay + dikey cevirme ortalamasi
                for dims in ([3], [2]):
                    lg, _ = model(torch.flip(images, dims=dims))
                    p = p + torch.sigmoid(torch.flip(lg.float(), dims=dims))
                p = p / 3.0
        probs_all.append(p.half().cpu())
        masks_all.append(masks.half())
    return torch.cat(probs_all), torch.cat(masks_all)


def tune_postprocess(probs, masks, thresholds, min_sizes):
    """Validation uzerinde en iyi (esik, min_size) ciftini ara.
    Neden? Varsayilan 0.5 esigi dengesiz veride nadiren optimaldir ve
    min_size filtresi kucuk yanlis pozitifleri temizler."""
    best, rows = None, []
    for th in thresholds:
        for ms in min_sizes:
            d = dice_per_sample(probs.float(), masks.float(), th, ms).mean()
            rows.append({"threshold": th, "min_size": ms, "val_dice": float(d)})
            if best is None or d > best[0]:
                best = (float(d), th, ms)
    return best, pd.DataFrame(rows)


def plot_confusion(y_true, y_pred, out_png, num_classes=4):
    """Her sinif icin 2x2 karisiklik matrisi (kusur var / yok tespiti)."""
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 3.6))
    for c in range(num_classes):
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])
        ax = axes[c]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"Class {c+1}")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def plot_examples(dataset, probs, masks, out_png, threshold, min_size, n=4):
    """Ornek tahminler: girdi + gercek maske + tahmin (hata analizi icin)."""
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 128, 255], [255, 255, 0]])
    # Kusur iceren ornekleri sec
    has = (masks.float().sum(dim=(2, 3)) > 0).any(dim=1).numpy()
    idxs = list(np.where(has)[0][:n]) or list(range(min(n, len(dataset))))

    fig, axes = plt.subplots(len(idxs), 1, figsize=(16, 2.6 * len(idxs)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idxs):
        import cv2
        row = dataset.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(dataset.img_dir, row["ImageId"])),
                           cv2.COLOR_BGR2RGB)
        gt = masks[i].float().numpy()
        pr = (probs[i].float().numpy() > threshold).astype(np.float32)
        for c in range(pr.shape[0]):
            if pr[c].sum() < min_size:
                pr[c] = 0
        vis = img.copy()
        for c in range(gt.shape[0]):
            vis[gt[c] > 0] = (0.45 * vis[gt[c] > 0] + 0.55 * colors[c]).astype(np.uint8)
            edge = pr[c] - np.minimum(pr[c], gt[c])   # sadece tahmin (yanlis pozitif) alanlar
            vis[edge > 0] = (0.35 * vis[edge > 0] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
        ax.imshow(vis); ax.axis("off")
        ax.set_title(f"{row['ImageId']} | renkli=gercek maske, macenta=fazladan tahmin", fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--tta", type=int, default=1)
    args = parser.parse_args()
    cfg = config_from_args(args)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, cfg.run_name)
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Egitimdeki config'i geri yukle (mimari ayni olmali!)
    saved = Config(**ckpt["cfg"])
    saved.data_dir, saved.out_dir, saved.run_name = cfg.data_dir, cfg.out_dir, cfg.run_name
    cfg = saved

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Model yuklendi: epoch {ckpt['epoch']}, val dice {ckpt['val_dice']:.4f}")

    # Egitimdeki AYNI bolmeleri kullan
    df = build_dataframe(cfg.data_dir, cfg.num_classes)
    from torch.utils.data import DataLoader
    loaders, datasets = {}, {}
    for split in ["val", "test"]:
        ids = pd.read_csv(os.path.join(out_dir, f"split_{split}.csv"))["ImageId"]
        sub = df[df["ImageId"].isin(set(ids))].reset_index(drop=True)
        ds = SteelDataset(sub, cfg.data_dir, cfg, train=False)
        datasets[split] = ds
        loaders[split] = DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False,
                                    num_workers=cfg.num_workers)

    # ---------- 1) Validation uzerinde son islem ayari ----------
    vp, vm = collect_predictions(model, loaders["val"], device, tta=bool(args.tta))
    (best_dice, best_th, best_ms), grid = tune_postprocess(
        vp, vm, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7], min_sizes=[0, 300, 600, 1200, 2000])
    grid.to_csv(os.path.join(out_dir, "postprocess_grid.csv"), index=False)
    print(f"En iyi son islem -> threshold={best_th}, min_size={best_ms}, val dice={best_dice:.4f}")

    # ---------- 2) Test metrikleri ----------
    tp, tm = collect_predictions(model, loaders["test"], device, tta=bool(args.tta))
    tracker = MetricTracker(cfg.num_classes)
    tracker.update(tp.float(), tm.float(), best_th, best_ms)
    metrics, (y_true, y_pred) = tracker.compute()

    # Varsayilan ayar ile karsilastirma (son islemin katkisini gostermek icin)
    base_tracker = MetricTracker(cfg.num_classes)
    base_tracker.update(tp.float(), tm.float(), 0.5, 0)
    base_metrics, _ = base_tracker.compute()

    # ---------- 3) Trivial baseline: "hep bos tahmin et" ----------
    empty_dice = float(dice_per_sample(torch.zeros_like(tp.float()), tm.float(), 0.5, 0).mean())

    metrics.update({"threshold": best_th, "min_size": best_ms,
                    "tta": bool(args.tta),
                    "dice_default_th0.5_ms0": base_metrics["mean_dice"],
                    "baseline_all_empty_dice": empty_dice,
                    "best_val_dice": float(ckpt["val_dice"]), "best_epoch": int(ckpt["epoch"])})
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------- 4) Gorseller ----------
    plot_confusion(y_true, y_pred, os.path.join(out_dir, "confusion_matrix.png"), cfg.num_classes)
    plot_examples(datasets["test"], tp, tm, os.path.join(out_dir, "predictions.png"),
                  best_th, best_ms)

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
