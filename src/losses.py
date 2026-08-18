"""Kayip fonksiyonlari.

Neden sadece BCE yetmiyor?
  Piksellerin ~%98'i arka plan. BCE bu dengesizlikte "her seye 0 de" cozumune
  yakinsayabilir. Dice kaybi ise dogrudan ortusme oranini optimize ettigi icin
  kucuk pozitif bolgeleri onemser. Ikisini birlestiriyoruz:
      L_seg = (1 - w) * BCE + w * SoftDice
  Ayrica yardimci siniflandirma kaybi ekleniyor (coklu gorev ogrenme):
      L_total = L_seg + lambda * BCE_cls
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Turevlenebilir Dice. Logit alir, iceride sigmoid uygular."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)  # batch ve uzamsal boyutlar uzerinde topla, sinif bazinda kalsin
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Kolay orneklerin katkisini bastirir (alternatif deney icin)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, cls_weight=0.3, pos_weight=2.0, use_focal=False):
        super().__init__()
        self.dice_weight = dice_weight
        self.cls_weight = cls_weight
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss() if use_focal else None
        self.register_buffer("pw", torch.tensor(pos_weight))

    def forward(self, seg_logits, seg_targets, cls_logits=None, cls_targets=None):
        if self.focal is not None:
            pixel = self.focal(seg_logits, seg_targets)
        else:
            pixel = F.binary_cross_entropy_with_logits(
                seg_logits, seg_targets, pos_weight=self.pw)
        dice = self.dice(seg_logits, seg_targets)
        loss = (1 - self.dice_weight) * pixel + self.dice_weight * dice

        parts = {"pixel": pixel.detach(), "dice": dice.detach()}
        if cls_logits is not None and self.cls_weight > 0:
            cls = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
            loss = loss + self.cls_weight * cls
            parts["cls"] = cls.detach()
        return loss, parts


def build_loss(cfg):
    return CombinedLoss(
        dice_weight=float(cfg.dice_weight),
        cls_weight=float(cfg.cls_loss_weight),
        pos_weight=float(cfg.pos_weight),
    )
