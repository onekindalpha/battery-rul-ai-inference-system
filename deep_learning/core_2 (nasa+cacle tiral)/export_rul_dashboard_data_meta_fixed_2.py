# deep_learning/core_2/export_rul_dashboard_data_meta_fixed_2.py

import os
import json
import argparse
import numpy as np
import sys
from pathlib import Path

from deep_learning.core_2.prefix_inference_viz_2 import (
    build_model_and_grouped,
    make_task_prefix,
    run_adapt_and_predict,
)

def export_bmaml_dashboard_json(
    ckpt_path: str,
    eval_dataset: str,
    r_ratio: float,
    out_dir: str,
    bids: str = "",
):
    os.makedirs(out_dir, exist_ok=True)

    (
        cfg,
        grouped,
        model,
        vecizer,
        meta_thetas,
        seq_scaler,
        sum_scaler,
        max_rul_train,
        ckpt_test_bids,
    ) = build_model_and_grouped(ckpt_path, eval_dataset)

    # 사용할 배터리 선택
    if bids:
        cand = [b.strip() for b in bids.split(",") if b.strip()]
        target_bids = [b for b in cand if b in grouped]
    else:
        cand = [b for b in ckpt_test_bids if b in grouped]
        if cand:
            target_bids = cand
        else:
            default_bids = ["B0018", "B0033", "B0042", "B0043"]
            target_bids = [b for b in default_bids if b in grouped]

    if not target_bids:
        raise RuntimeError("No valid target batteries for BMAML export.")

    print(f"[BMAML EXPORT] Using batteries: {target_bids}")

    for bid in target_bids:
        print(f"[BMAML EXPORT] {bid} ...")

        task = make_task_prefix(
            bid,
            grouped,
            cfg,
            seq_scaler,
            sum_scaler,
            max_rul_train,
            r_ratio=r_ratio,
            current_cycle=None,
        )

        # pred_mean, pred_std : query 구간에 대한 예측 + 불확실성
        pred_mean, pred_std = run_adapt_and_predict(
            cfg, model, vecizer, meta_thetas, task, max_rul_train
        )
        if pred_mean is None:
            print(f"[WARN] BMAML inference failed for {bid}, skip.")
            continue

        s_cyc = np.asarray(task["s_cycles_viz"], dtype=float)
        s_true = np.asarray(task["s_rul_viz"], dtype=float)
        q_cyc = np.asarray(task["q_cycles_viz"], dtype=float)
        q_true = np.asarray(task["q_rul_viz"], dtype=float)
        split_cycle = float(task["split_cycle"])

        pred_mean = np.asarray(pred_mean, dtype=float)
        pred_std = np.asarray(pred_std, dtype=float)

        # ---- 전체 시계열 구성 ----
        cycles_full = np.concatenate([s_cyc, q_cyc])
        rul_true_full = np.concatenate([s_true, q_true])

        # history 구간 예측/표준편차는 NaN으로 채워서 plot에서 안 그려지게 함
        nan_hist = np.full_like(s_true, np.nan, dtype=float)
        pred_full = np.concatenate([nan_hist, pred_mean])
        std_full = np.concatenate([nan_hist, pred_std])

        # RMSE / MAE는 q 구간에서만 계산
        rmse = float(np.sqrt(np.mean((q_true - pred_mean) ** 2)))
        mae = float(np.mean(np.abs(q_true - pred_mean)))

        rec = {
            "battery_id": bid,
            "model_type": "bmaml_svgd",
            "cycles": cycles_full.tolist(),
            "rul_true": rul_true_full.tolist(),
            "rul_pred": pred_full.tolist(),
            "rul_std": std_full.tolist(),
            "split_cycle": split_cycle,
            "rmse": rmse,
            "mae": mae,
        }

        out_path = os.path.join(out_dir, f"battery_{bid}.json")
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[BMAML EXPORT] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="BMAML checkpoint path"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="from_ckpt",
        choices=["from_ckpt", "nasa", "cacle", "both"],
    )
    parser.add_argument("--r_ratio", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--bids",
        type=str,
        default="",
        help="comma separated battery ids (optional)",
    )
    args = parser.parse_args()

    export_bmaml_dashboard_json(
        ckpt_path=args.ckpt,
        eval_dataset=args.eval_dataset,
        r_ratio=args.r_ratio,
        out_dir=args.out_dir,
        bids=args.bids,
    )


if __name__ == "__main__":
    main()


# python -m deep_learning.core_2.export_rul_dashboard_data_meta_fixed_2 \
#   --ckpt deep_learning/core_2/core_checkpoints/cacle_bmaml_best_re_.pt \
#   --eval_dataset from_ckpt \
#   --r_ratio 0.3 \
#   --bids CS2_35,CS2_36,CS2_37,CS2_38 \
#   --out_dir deep_learning/core_2/dashboard_export/bmaml_cacle
