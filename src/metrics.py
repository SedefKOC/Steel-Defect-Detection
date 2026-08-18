"""Metrikler.

Dice katsayisi (Kaggle kurali):
    Bir (goruntu, sinif) cifti icin
      - hem gercek hem tahmin BOSSA  -> dice = 1.0
      - biri bos digeri doluysa      -> dice = 0.0
      - ikisi de doluysa             -> 2|X n Y| / (|X| + |Y|)
    Nihai skor = tum (goruntu, sinif) ciftlerinin ortalamasi.

Neden accuracy kullanmiyoruz? Piksellerin %98'i arka plan; hicbir sey tahmin
etmeyen model %98 accuracy alir ama tamamen ise yaramazdir.
"""
import numpy as np
import torch


@torch.no_grad()
def dice_per_sample(probs, targets, threshold=0.5, min_size=0):
    """probs: (B,C,H,W) sigmoid sonrasi, targets: (B,C,H,W) 0/1.
    Donen: (B,C) dice matrisi (numpy)."""
    preds = (probs > threshold).float()

    if min_size > 0:  # kucuk lekeleri sil (yanlis pozitif azaltma)
        areas = preds.sum(dim=(2, 3), keepdim=True)
        preds = preds * (areas >= min_size).float()

    inter = (preds * targets).sum(dim=(2, 3))
    p_sum = preds.sum(dim=(2, 3))
    t_sum = targets.sum(dim=(2, 3))

    dice = torch.where(
        (p_sum + t_sum) == 0,
        torch.ones_like(p_sum),                      # ikisi de bos -> 1.0
        (2 * inter) / (p_sum + t_sum + 1e-7))
    return dice.detach().cpu().numpy()


@torch.no_grad()
def iou_per_sample(probs, targets, threshold=0.5):
    preds = (probs > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(2, 3))
    iou = torch.where(union == 0, torch.ones_like(union), inter / (union + 1e-7))
    return iou.detach().cpu().numpy()


class MetricTracker:
    """Epoch boyunca (goruntu, sinif) bazli dice degerlerini biriktirir."""

    def __init__(self, num_classes=4):
        self.nc = num_classes
        self.reset()

    def reset(self):
        self.dices = []
        self.ious = []
        self.pred_labels = []
        self.true_labels = []

    def update(self, probs, targets, threshold=0.5, min_size=0):
        self.dices.append(dice_per_sample(probs, targets, threshold, min_size))
        self.ious.append(iou_per_sample(probs, targets, threshold))
        preds = (probs > threshold).float()
        self.pred_labels.append((preds.sum(dim=(2, 3)) > min_size).float().cpu().numpy())
        self.true_labels.append((targets.sum(dim=(2, 3)) > 0).float().cpu().numpy())

    def compute(self):
        d = np.concatenate(self.dices, axis=0)     # (N, C)
        i = np.concatenate(self.ious, axis=0)
        yp = np.concatenate(self.pred_labels, axis=0)
        yt = np.concatenate(self.true_labels, axis=0)

        res = {
            "mean_dice": float(d.mean()),
            "mean_iou": float(i.mean()),
        }
        for c in range(self.nc):
            res[f"dice_class{c+1}"] = float(d[:, c].mean())
            pos = yt[:, c] == 1
            # Sadece gercekten kusur iceren orneklerdeki dice (daha zorlu, daha bilgilendirici)
            res[f"dice_pos_class{c+1}"] = float(d[pos, c].mean()) if pos.sum() else float("nan")

        # Sinif bazli tespit metrikleri (precision / recall / F1)
        try:
            from sklearn.metrics import precision_recall_fscore_support
            p, r, f1, _ = precision_recall_fscore_support(
                yt, yp, average=None, zero_division=0, labels=list(range(self.nc)))
            for c in range(self.nc):
                res[f"precision_class{c+1}"] = float(p[c])
                res[f"recall_class{c+1}"] = float(r[c])
                res[f"f1_class{c+1}"] = float(f1[c])
            res["macro_f1"] = float(np.mean(f1))
        except Exception:
            pass
        return res, (yt, yp)
