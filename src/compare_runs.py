"""Tum kosulari tek bir ablasyon tablosunda toplar (rapordaki 5.2 bolumu icin).

Kullanim:
    python src/compare_runs.py --out_dir outputs
"""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    rows = []
    for run in sorted(os.listdir(args.out_dir)):
        d = os.path.join(args.out_dir, run)
        s_path, m_path = os.path.join(d, "train_summary.json"), os.path.join(d, "metrics_test.json")
        if not os.path.exists(s_path):
            continue
        s = json.load(open(s_path))
        c = s["config"]
        row = {
            "run": run,
            "pretrained": c["pretrained"], "BN": c["use_bn"],
            "dropout": c["decoder_dropout"], "attention": c["use_attention"],
            "augment": c["augment"], "weight_decay": c["weight_decay"],
            "lr": c["lr"], "scheduler": c["scheduler"], "activation": c["activation"],
            "cls_w": c["cls_loss_weight"],
            "epochs_run": s["epochs_run"], "best_epoch": s["best_epoch"],
            "val_dice": round(s["best_val_dice"], 4), "minutes": s["minutes"],
        }
        if os.path.exists(m_path):
            m = json.load(open(m_path))
            row["test_dice"] = round(m["mean_dice"], 4)
            row["test_iou"] = round(m["mean_iou"], 4)
        rows.append(row)

    if not rows:
        print(f"'{args.out_dir}' altinda tamamlanmis kosu bulunamadi.")
        return

    df = pd.DataFrame(rows).sort_values("val_dice", ascending=False)
    df.to_csv(os.path.join(args.out_dir, "ablation_table.csv"), index=False)
    md = df.to_markdown(index=False)
    with open(os.path.join(args.out_dir, "ablation_table.md"), "w") as f:
        f.write("## Ablasyon / deney karsilastirmasi\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
