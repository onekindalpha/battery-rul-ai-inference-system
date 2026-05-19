# -*- coding: utf-8 -*-
"""
prefix_inference_viz_backbone.py (pyc-aligned restore)

This version is intentionally structured to be closer to the original CPython-3.10 .pyc:
- Keeps the top-level function names seen in the .pyc:
  load_cfg_from_ckpt, build_model_and_grouped_backbone, make_task_prefix,
  run_predict_backbone, plot_improved_curve, main
- Avoids extra helper function names that did not appear in the .pyc.

Backbone-only prefix inference + visualization (no inner-loop adaptation).
"""

import os
import argparse

import numpy as np
import torch

# plotting (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401
    sns.set_style("whitegrid")
except Exception:
    plt = None

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.models import MultiTaskRULModel
from deep_learning.core.meta_utils import DEVICE
from deep_learning.core.scalers import unscale_rul_array


def load_cfg_from_ckpt(cfg_dict):
    cfg = Config()
    for k, v in (cfg_dict or {}).items():
        setattr(cfg, k, v)
    return cfg


def build_model_and_grouped_backbone(ckpt_path, eval_dataset="from_ckpt"):
    print("[LOAD] Backbone checkpoint: " + str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = load_cfg_from_ckpt(ckpt.get("config", {}))

    feature_cols = ckpt.get("feature_cols", None)
    target_col = ckpt.get("target_col", None)

    eval_dataset = (eval_dataset or "from_ckpt").lower()
    if eval_dataset != "from_ckpt":
        cfg.dataset_source = eval_dataset
        print("[EVAL] cross-domain on dataset_source='" + str(cfg.dataset_source) + "'")
    else:
        print("[EVAL] dataset_source from ckpt: " + str(getattr(cfg, "dataset_source", None)))

    df, tgt, feat, bid_col, cyc_col = load_cycle_db_for_meta(cfg)
    if target_col is None:
        target_col = tgt
    if feature_cols is None:
        feature_cols = feat

    grouped = group_data_by_battery_from_df(
        df=df,
        cfg=cfg,
        target_col=target_col,
        feature_cols=feature_cols,
        bid_col=bid_col,
        cyc_col=cyc_col,
    )

    # ckpt scalers / max_rul_train (expected for backbone ckpt)
    seq_scaler = ckpt["seq_scaler"]
    sum_scaler = ckpt["sum_scaler"]
    max_rul_train = float(ckpt["max_rul_train"])

    # weights key (expected)
    state = ckpt["model_state"]

    # infer dims inline (avoid extra helper name)
    if getattr(cfg, "use_resnet", False) and ("resnet.stem.weight" in state):
        sd = int(state["resnet.stem.weight"].shape[1])
    else:
        sd = int(state["proj_seq.weight"].shape[1])
    sm = int(state["pm.weight"].shape[1])

    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    test_bids = ckpt.get("test_bids", sorted(grouped.keys()))
    test_bids = [b for b in test_bids if b in grouped]
    print("[INFO] Eval Batteries (" + str(len(test_bids)) + "): " + str(test_bids))

    return cfg, grouped, model, seq_scaler, sum_scaler, max_rul_train, test_bids


def make_task_prefix(bid, grouped, cfg, seq_scaler, sum_scaler, max_rul_train, r_ratio=0.5, current_cycle=None):
    g = grouped[bid]

    seq_w = np.asarray(g["seq"], dtype=np.float32)
    sum_w = np.asarray(g["sum"], dtype=np.float32)
    rul_raw = np.asarray(g["rul"], dtype=np.float32)

    T = int(len(rul_raw))
    if T < 2:
        raise ValueError("Not enough windows for bid=" + str(bid) + " (T=" + str(T) + ")")

    # align cycles to window-end cycles (inline)
    cycles_raw = np.asarray(g.get("cycle", np.arange(T, dtype=np.float32)), dtype=np.float32)
    seq_len = int(getattr(cfg, "seq_len", 1))
    start = max(seq_len - 1, 0)
    win_cycles = cycles_raw[start : start + T]
    if len(win_cycles) != T:
        m = min(len(win_cycles), T)
        seq_w, sum_w, rul_raw, win_cycles = seq_w[:m], sum_w[:m], rul_raw[:m], win_cycles[:m]
        T = m

    # decide split
    if current_cycle is not None:
        idx = np.where(win_cycles >= float(current_cycle))[0]
        prefix_len = int(idx[0] + 1) if len(idx) else (T - 1)
        r_ratio_input = None
    else:
        prefix_len = int(float(r_ratio) * float(T))
        r_ratio_input = float(r_ratio)

    prefix_len = int(max(1, min(T - 1, prefix_len)))

    # scale query inputs
    q_idx = np.arange(prefix_len, T, dtype=int)

    q_seq_sc = seq_scaler.transform(seq_w[q_idx])
    q_sum_sc = sum_scaler.transform(sum_w[q_idx])

    task = {
        "battery_id": bid,
        "win_cycles": np.asarray(win_cycles, dtype=np.float32),
        "rul_raw": np.asarray(rul_raw, dtype=np.float32),
        "prefix_len": int(prefix_len),
        "q_seq": torch.from_numpy(np.asarray(q_seq_sc)).float(),
        "q_sum": torch.from_numpy(np.asarray(q_sum_sc)).float(),
        "r_ratio_input": r_ratio_input,
        "r_ratio_effective": float(prefix_len) / float(T) if T > 0 else None,
        "split_cycle": float(win_cycles[prefix_len - 1]),
    }
    return task


def run_predict_backbone(cfg, model, task, max_rul_train, mc_samples=20):
    q_seq = torch.nan_to_num(task["q_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
    q_sum = torch.nan_to_num(task["q_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

    preds = []
    model.train()  # enable dropout for MC
    with torch.no_grad():
        for _ in range(int(mc_samples)):
            out = model(q_seq, q_sum)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds.append(out.squeeze(-1).detach().cpu().numpy())
    model.eval()

    preds = np.stack(preds, axis=0)
    mean_sc = preds.mean(axis=0)
    std_sc = preds.std(axis=0)

    pred_mean = unscale_rul_array(mean_sc, getattr(cfg, "rul_mode", "minmax"), max_rul_train)
    upper_sc = mean_sc + std_sc
    upper_real = unscale_rul_array(upper_sc, getattr(cfg, "rul_mode", "minmax"), max_rul_train)
    pred_std = np.abs(upper_real - pred_mean)

    return np.asarray(pred_mean, dtype=float), np.asarray(pred_std, dtype=float)


def plot_improved_curve(bid, task, pred_mean, pred_std, out_dir):
    if plt is None:
        return

    win_cycles = np.asarray(task["win_cycles"], dtype=float)
    rul_raw = np.asarray(task["rul_raw"], dtype=float)
    prefix_len = int(task["prefix_len"])
    split_cycle = float(task["split_cycle"])

    s_cyc = win_cycles[:prefix_len]
    s_true = rul_raw[:prefix_len]
    q_cyc = win_cycles[prefix_len:]
    q_true = rul_raw[prefix_len:]

    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)

    m = min(len(q_cyc), len(pred_mean))
    q_cyc, q_true, pred_mean, pred_std = q_cyc[:m], q_true[:m], pred_mean[:m], pred_std[:m]

    lower = np.maximum(0.0, pred_mean - 2.0 * pred_std)
    upper = pred_mean + 2.0 * pred_std

    rmse = float(np.sqrt(np.mean((pred_mean - q_true) ** 2))) if len(q_true) else float("nan")
    mae = float(np.mean(np.abs(pred_mean - q_true))) if len(q_true) else float("nan")

    plt.figure(figsize=(14, 10), dpi=150)
    plt.plot(s_cyc, s_true, linewidth=1.7, label="Observed History")
    plt.plot(q_cyc, q_true, linestyle="--", linewidth=1.5, label="True Future RUL")
    plt.plot(q_cyc, pred_mean, linewidth=1.8, label="Predicted RUL")
    plt.fill_between(q_cyc, lower, upper, alpha=0.25, label="Uncertainty (±2σ)")
    plt.axvline(split_cycle, linestyle=":", linewidth=1.5)

    title = f"{bid} | BACKBONE | RMSE={rmse:.3f}, MAE={mae:.3f}"
    if task.get("r_ratio_effective", None) is not None:
        title += f" (r_ratio={float(task['r_ratio_effective']):.2f})"
    plt.title(title)
    plt.xlabel("cycle")
    plt.ylabel("RUL")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{bid}_backbone_viz.png")
    plt.savefig(path)
    print("[PLOT] Saved to " + str(path))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, default="from_ckpt")
    parser.add_argument("--bids", type=str, default="")
    parser.add_argument("--r_ratio", type=float, default=0.5)
    parser.add_argument("--current_cycle", type=float, default=None)
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="./viz_backbone")

    args = parser.parse_args()

    cfg, grouped, model, seq_scaler, sum_scaler, max_rul_train, test_bids = build_model_and_grouped_backbone(
        args.ckpt, args.eval_dataset
    )

    if str(args.bids).strip():
        cand = [b.strip() for b in str(args.bids).split(",") if b.strip()]
        target_bids = [b for b in cand if b in grouped]
        missing = [b for b in cand if b not in grouped]
        if missing:
            print("[WARN] Some requested bids not in grouped: " + str(missing))
    else:
        target_bids = test_bids

    if not target_bids:
        print("[ERROR] No valid target batteries found.")
        return

    print("[INFO] Using target batteries: " + str(target_bids))
    os.makedirs(args.out_dir, exist_ok=True)

    for bid in target_bids:
        try:
            task = make_task_prefix(
                bid,
                grouped,
                cfg,
                seq_scaler,
                sum_scaler,
                max_rul_train,
                r_ratio=args.r_ratio,
                current_cycle=args.current_cycle,
            )
        except Exception as e:
            print("[SKIP] " + str(bid) + " (" + str(e) + ")")
            continue

        pred_mean, pred_std = run_predict_backbone(cfg, model, task, max_rul_train, args.mc_samples)
        plot_improved_curve(bid, task, pred_mean, pred_std, args.out_dir)


if __name__ == "__main__":
    main()
