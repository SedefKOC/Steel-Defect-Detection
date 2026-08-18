"""Egitim scripti.

Kullanim:
    python src/train.py --run_name final
    python src/train.py --run_name baseline --pretrained 0 --use_bn 0 --augment 0
    python src/train.py --limit 300 --epochs 2 --run_name smoke_test    # hizli test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, add_config_args, config_from_args
from dataset import make_loaders
from losses import build_loss
from metrics import MetricTracker
from model import build_model, count_parameters
from utils import AverageMeter, plot_history, set_seed


# --------------------------------------------------------------------------
def build_optimizer(model, cfg):
    """Optimizer secimi.
    AdamW: Adam'in weight decay'i gradyandan ayirdigi (decoupled) versiyonu.
    L2 duzenlilestirmenin adaptif optimizerlarla dogru calismasini saglar."""
    params = model.parameters()
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999),
                                 weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(cfg.optimizer)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.scheduler == "cosine":
        # Kosinus tavlama: LR yumusakca 0'a iner, sonlara dogru ince ayar yapar
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01), "epoch"
    if cfg.scheduler == "plateau":
        # Val dice 3 epoch iyilesmezse LR'yi yariya indir
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3), "plateau"
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3), "epoch"
    return None, "none"


# --------------------------------------------------------------------------
class EarlyStopping:
    """Validation dice belirli sayida epoch iyilesmezse egitimi durdurur.
    Gereksiz epoch harcamayi ve overfitting'i engeller."""

    def __init__(self, patience=6, mode="max", min_delta=1e-4):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best = -np.inf if mode == "max" else np.inf
        self.counter, self.should_stop, self.best_epoch = 0, False, 0

    def step(self, value, epoch):
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, cfg, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    loss_meter = AverageMeter()
    tracker = MetricTracker(cfg.num_classes)
    amp = bool(int(cfg.amp)) and device.type == "cuda"

    for images, masks, labels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp):
                seg_logits, cls_logits = model(images)
                loss, _ = criterion(seg_logits, masks, cls_logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        with torch.no_grad():
            tracker.update(torch.sigmoid(seg_logits.float()), masks, cfg.threshold, 0)

    metrics, _ = tracker.compute()
    return loss_meter.avg, metrics


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args())

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"RUN: {cfg.run_name}  |  device: {device}")
    print("=" * 70)

    train_loader, val_loader, test_loader, dfs = make_loaders(cfg)
    train_df, val_df, test_df = dfs
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print("Sinif dagilimi (train):",
          {f"class{c}": int(train_df[f'has{c}'].sum()) for c in range(1, cfg.num_classes + 1)},
          f"kusursuz={int((train_df['n_defects'] == 0).sum())}")

    model = build_model(cfg).to(device)
    total, trainable = count_parameters(model)
    print(f"parametre: {total/1e6:.2f}M (egitilebilir {trainable/1e6:.2f}M)")

    criterion = build_loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler, sched_type = build_scheduler(optimizer, cfg, len(train_loader))
    use_amp = bool(int(cfg.amp)) and device.type == "cuda"
    try:                                   # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):    # eski surumler
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    stopper = EarlyStopping(patience=cfg.early_stop_patience, mode="max")

    history, best_path = [], os.path.join(out_dir, "best_model.pt")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, cfg, device, optimizer, scaler)
        va_loss, va_m = run_epoch(model, val_loader, criterion, cfg, device)

        if sched_type == "plateau":
            scheduler.step(va_m["mean_dice"])
        elif sched_type == "epoch":
            scheduler.step()

        improved = stopper.step(va_m["mean_dice"], epoch)
        if improved:
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                        "epoch": epoch, "val_dice": va_m["mean_dice"]}, best_path)

        history.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr_loss, "val_loss": va_loss,
            "train_dice": tr_m["mean_dice"], "val_dice": va_m["mean_dice"],
            "val_iou": va_m["mean_iou"],
            **{f"val_{k}": v for k, v in va_m.items() if k.startswith("dice_class")},
        })
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

        flag = "  <-- best (kaydedildi)" if improved else f"  (sabir {stopper.counter}/{cfg.early_stop_patience})"
        print(f"epoch {epoch:03d}/{cfg.epochs} | lr {lr_now:.2e} | "
              f"train loss {tr_loss:.4f} dice {tr_m['mean_dice']:.4f} | "
              f"val loss {va_loss:.4f} dice {va_m['mean_dice']:.4f}{flag}")

        if stopper.should_stop:
            print(f"\n[EARLY STOPPING] {epoch}. epoch'ta durduruldu. "
                  f"En iyi epoch: {stopper.best_epoch} (val dice {stopper.best:.4f})")
            break

    mins = (time.time() - t0) / 60
    plot_history(os.path.join(out_dir, "history.csv"), os.path.join(out_dir, "curves.png"))

    # Bolme dosyalarini kaydet -> evaluate.py ayni test setini kullansin
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d[["ImageId"]].to_csv(os.path.join(out_dir, f"split_{name}.csv"), index=False)

    summary = {
        "run_name": cfg.run_name, "config": cfg.to_dict(),
        "best_epoch": stopper.best_epoch, "best_val_dice": float(stopper.best),
        "epochs_run": len(history), "minutes": round(mins, 1),
        "params_million": round(total / 1e6, 2), "device": str(device),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBitti ({mins:.1f} dk). En iyi val dice: {stopper.best:.4f} "
          f"(epoch {stopper.best_epoch})\nCiktilar: {out_dir}")


if __name__ == "__main__":
    main()
