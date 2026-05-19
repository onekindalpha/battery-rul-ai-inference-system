# -*- coding: utf-8 -*-
"""
prefix_inference_viz_2.py (pyc-aligned restore)

This version is intentionally structured to be closer to the original CPython-3.10 .pyc:
- Keeps the top-level function names seen in the .pyc:
  load_cfg_from_ckpt, build_model_and_grouped, make_task_prefix,
  run_adapt_and_predict, plot_improved_curve, main
- Avoids extra helper function names that did not appear in the .pyc.

META(BMAML-SVGD) prefix inference + visualization.

NOTE: This "pyc-aligned" variant imports deep_learning.core_2.*_2 modules directly (no fallback),
because the original .pyc came from your core_2 path.
"""

import os
import argparse

import numpy as np
import torch
from torch.nn.utils.stateless import functional_call

# plotting (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401
    sns.set_style("whitegrid")
except Exception:
    plt = None

from sklearn.preprocessing import RobustScaler

# core_2 imports (as in the original pyc context)
from deep_learning.core_2.config_2 import Config
from deep_learning.core_2.meta_db_loader_2 import load_cycle_db_for_meta
from deep_learning.core_2.data_seq_group_2 import group_data_by_battery_from_df
from deep_learning.core_2.models_2 import MultiTaskRULModel
from deep_learning.core_2.meta_utils_2 import DEVICE, ParamVectorizer, bmaml_inner, make_leaf_thetas
from deep_learning.core_2.scalers_2 import CustomRobustScaler3D, unscale_rul_array, scale_rul_array


def load_cfg_from_ckpt(cfg_dict):
    cfg = Config()
    for k, v in (cfg_dict or {}).items():
        setattr(cfg, k, v)
    return cfg


def build_model_and_grouped(ckpt_path, eval_dataset="from_ckpt"):
    print("[LOAD] ckpt: " + str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = load_cfg_from_ckpt(ckpt.get("config", {}))

    feature_cols = ckpt.get("feature_cols", None)
    target_col = ckpt.get("target_col", None)

    eval_dataset = (eval_dataset or "from_ckpt").lower()
    if eval_dataset != "from_ckpt":
        cfg.dataset_source = eval_dataset
        print("[EVAL] cross-domain eval: dataset_source (" + str(cfg.dataset_source) + ")")
    else:
        print("[EVAL] dataset_source (" + str(getattr(cfg, "dataset_source", None)) + ")")

    df, tgt, feat, bid_col, cyc_col = load_cycle_db_for_meta(cfg)
    if target_col is None:
        target_col = tgt
    if feature_cols is None:
        feature_cols = feat

    grouped = group_data_by_battery_from_df(
        df=df, cfg=cfg, target_col=target_col, feature_cols=feature_cols, bid_col=bid_col, cyc_col=cyc_col
    )

    # scalers (expected for BMAML ckpt; keep a simple fallback)
    if all(k in ckpt for k in ("seq_scaler", "sum_scaler", "max_rul_train")):
        seq_scaler = ckpt["seq_scaler"]
        sum_scaler = ckpt["sum_scaler"]
        max_rul_train = float(ckpt["max_rul_train"])
        print("[INIT] config / scaler / max_rul_train loaded from ckpt")
    else:
        print("[WARN] ckpt has no scaler/max_rul_train. Fitting fallback.")
        seq_all = np.concatenate([np.asarray(grouped[b]["seq"], dtype=np.float32) for b in grouped.keys()], axis=0)
        sum_all = np.concatenate([np.asarray(grouped[b]["sum"], dtype=np.float32) for b in grouped.keys()], axis=0)

        seq_scaler = CustomRobustScaler3D(p=90, min_iqr=1e-3, clip_value=float(getattr(cfg, "seq_clip", 10.0)))
        seq_scaler.fit(seq_all)

        sum_scaler = RobustScaler()
        if sum_all.ndim == 1:
            sum_all = sum_all.reshape(-1, 1)
        sum_scaler.fit(sum_all)

        max_rul_train = float(max(float(grouped[b]["max_rul"]) for b in grouped.keys()))

    # model + meta particles
    state = ckpt["model_state"]

    # infer dims inline
    if getattr(cfg, "use_resnet", False) and ("resnet.stem.weight" in state):
        sd = int(state["resnet.stem.weight"].shape[1])
    else:
        sd = int(state["proj_seq.weight"].shape[1])
    sm = int(state["pm.weight"].shape[1])

    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    if "meta_thetas" not in ckpt:
        raise KeyError("Checkpoint missing meta_thetas (this script expects a BMAML checkpoint).")
    meta_thetas = [t.to(DEVICE) for t in ckpt["meta_thetas"]]
    vecizer = ParamVectorizer(model)

    test_bids = ckpt.get("test_bids", sorted(grouped.keys()))
    test_bids = [b for b in test_bids if b in grouped]
    print("[INFO] grouped/test_bids " + str(len(test_bids)) + " : " + str(test_bids))

    return cfg, grouped, model, vecizer, meta_thetas, seq_scaler, sum_scaler, max_rul_train, test_bids


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

    # EOL index for reporting
    pos_idx = np.where(rul_raw > 0)[0]
    eol_idx = int(pos_idx[-1] + 1) if pos_idx.size > 0 else T

    # split
    if current_cycle is not None:
        idx = np.where(win_cycles >= float(current_cycle))[0]
        prefix_len = int(idx[0] + 1) if len(idx) else (T - 1)
        r_ratio_input = None
    else:
        prefix_len = int(float(r_ratio) * float(T))
        r_ratio_input = float(r_ratio)

    prefix_len = max(1, min(T - 1, prefix_len))
    prefix_len = min(prefix_len, eol_idx)

    split_cycle = float(win_cycles[prefix_len - 1])
    eol_ratio = float(prefix_len) / float(eol_idx) if eol_idx > 0 else None

    # support indices (inline)
    k = int(getattr(cfg, "k_shot", 16))
    if prefix_len <= k:
        s_idx = np.arange(prefix_len, dtype=int)
    else:
        strat = str(getattr(cfg, "support_strategy", "uniform")).lower()
        if strat == "mixed":
            half_k = k // 2
            recent_idx = np.arange(prefix_len - half_k, prefix_len, dtype=int)
            random_idx = np.sort(np.random.choice(prefix_len, size=k - half_k, replace=False)).astype(int)
            s_idx = np.unique(np.concatenate([recent_idx, random_idx]))[:k]
        elif strat == "recent":
            s_idx = np.arange(prefix_len - k, prefix_len, dtype=int)
        elif strat == "random":
            s_idx = np.sort(np.random.choice(prefix_len, size=k, replace=False)).astype(int)
        else:
            s_idx = np.linspace(0, prefix_len - 1, k, dtype=int)

    q_idx = np.arange(prefix_len, T, dtype=int)

    # scale inputs/targets
    s_seq_sc = seq_scaler.transform(seq_w[s_idx])
    q_seq_sc = seq_scaler.transform(seq_w[q_idx])

    s_sum_sc = sum_scaler.transform(sum_w[s_idx])
    q_sum_sc = sum_scaler.transform(sum_w[q_idx])

    s_rul_sc = scale_rul_array(rul_raw[s_idx], getattr(cfg, "rul_mode", "minmax"), max_rul_train)
    q_rul_sc = scale_rul_array(rul_raw[q_idx], getattr(cfg, "rul_mode", "minmax"), max_rul_train)

    cycles_union = np.concatenate([win_cycles[s_idx], win_cycles[q_idx]], axis=0).astype(np.float32)

    task = {
        "battery_id": bid,
        "s_seq": torch.from_numpy(np.asarray(s_seq_sc)).float(),
        "s_sum": torch.from_numpy(np.asarray(s_sum_sc)).float(),
        "s_rul": torch.from_numpy(np.asarray(s_rul_sc, dtype=np.float32)).float(),
        "q_seq": torch.from_numpy(np.asarray(q_seq_sc)).float(),
        "q_sum": torch.from_numpy(np.asarray(q_sum_sc)).float(),
        "q_rul": torch.from_numpy(np.asarray(q_rul_sc, dtype=np.float32)).float(),
        "cycles": torch.from_numpy(cycles_union).float(),
        "s_cycles_viz": np.asarray(win_cycles[s_idx], dtype=np.float32),
        "s_rul_viz": np.asarray(rul_raw[s_idx], dtype=np.float32),
        "q_cycles_viz": np.asarray(win_cycles[q_idx], dtype=np.float32),
        "q_rul_viz": np.asarray(rul_raw[q_idx], dtype=np.float32),
        "split_cycle": split_cycle,
        "r_ratio_input": r_ratio_input,
        "r_ratio_effective": float(prefix_len) / float(T) if T > 0 else None,
        "eol_ratio": eol_ratio,
        "current_true_rul": float(rul_raw[prefix_len - 1]),
        "current_cycle_effective": float(win_cycles[prefix_len - 1]),
    }
    return task


def run_adapt_and_predict(cfg, model, vecizer, meta_thetas, task, max_rul_train):
    with torch.enable_grad():
        theta0 = make_leaf_thetas(meta_thetas, detach=True)
        theta_p, _, eff, *_ = bmaml_inner(
            model,
            vecizer,
            theta0,
            task,
            cfg,
            detach_theta0=True,
            return_losses=False,
        )

    if (not eff) or (theta_p is None):
        return None, None

    q_seq = torch.nan_to_num(task["q_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
    q_sum = torch.nan_to_num(task["q_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

    preds = []
    with torch.no_grad():
        for tp in theta_p:
            params = vecizer.vector_to_params(tp)
            out = functional_call(model, params, (q_seq, q_sum))
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            preds.append(y_hat.squeeze(-1).detach().cpu().numpy())

    if not preds:
        return None, None

    preds_stack = np.stack(preds, axis=0)
    mean_sc = preds_stack.mean(axis=0)
    std_sc = preds_stack.std(axis=0)

    pred_mean = unscale_rul_array(mean_sc, getattr(cfg, "rul_mode", "minmax"), max_rul_train)

    upper_sc = mean_sc + std_sc
    upper_real = unscale_rul_array(upper_sc, getattr(cfg, "rul_mode", "minmax"), max_rul_train)
    pred_std = np.abs(upper_real - pred_mean)

    return np.asarray(pred_mean, dtype=float), np.asarray(pred_std, dtype=float)


def plot_improved_curve(bid, task, pred_mean, pred_std, out_dir):
    if plt is None:
        return

    s_cyc = np.asarray(task["s_cycles_viz"], dtype=float)
    s_true = np.asarray(task["s_rul_viz"], dtype=float)
    q_cyc = np.asarray(task["q_cycles_viz"], dtype=float)
    q_true = np.asarray(task["q_rul_viz"], dtype=float)
    split_pt = float(task["split_cycle"])

    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)

    m = min(len(q_cyc), len(pred_mean))
    q_cyc, q_true, pred_mean, pred_std = q_cyc[:m], q_true[:m], pred_mean[:m], pred_std[:m]

    lower = np.maximum(0.0, pred_mean - 2.0 * pred_std)
    upper = pred_mean + 2.0 * pred_std

    rmse = float(np.sqrt(np.mean((pred_mean - q_true) ** 2))) if len(q_true) else float("nan")
    mae = float(np.mean(np.abs(pred_mean - q_true))) if len(q_true) else float("nan")

    title = f"{bid} | RUL | RMSE={rmse:.3f}, MAE={mae:.3f}"
    if task.get("eol_ratio", None) is not None:
        title += f" (End of Life) NrR={float(task['eol_ratio']):.2f}"
    if task.get("r_ratio_effective", None) is not None:
        title += f" (r_ratio={float(task['r_ratio_effective']):.2f})"

    plt.figure(figsize=(14, 10), dpi=150)
    plt.plot(s_cyc, s_true, linewidth=1.8, label="Support(" + str(len(s_true)) + ")")
    plt.plot(q_cyc, q_true, linestyle="--", linewidth=1.8, label="True Future RUL")
    plt.plot(q_cyc, pred_mean, linewidth=1.8, label="Predicted RUL")
    plt.fill_between(q_cyc, lower, upper, alpha=0.25, label="Uncertainty (±2σ)")
    plt.axvline(split_pt, linestyle=":", linewidth=1.5, label="split")

    plt.xlabel("cycle")
    plt.ylabel("RUL")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{bid}_viz.png")
    plt.savefig(path)
    print("[PLOT] Saved to " + str(path))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--r_ratio", type=float, default=0.5)
    parser.add_argument("--current_cycle", type=float, default=None)
    parser.add_argument("--eval_dataset", type=str, default="from_ckpt", choices=["from_ckpt", "nasa", "cacle", "both"])
    parser.add_argument("--bids", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./viz_nasa_test_2")

    args = parser.parse_args()

    cfg, grouped, model, vecizer, meta_thetas, seq_scaler, sum_scaler, max_rul_train, ckpt_test_bids = build_model_and_grouped(
        args.ckpt, args.eval_dataset
    )

    # choose batteries
    if str(args.bids).strip():
        cand = [x.strip() for x in str(args.bids).split(",") if x.strip()]
        target_bids = [b for b in cand if b in grouped]
        missing = [b for b in cand if b not in grouped]
        if missing:
            print("[WARN] Some requested bids not in grouped: " + str(missing))
    else:
        target_bids = ckpt_test_bids

    if not target_bids:
        print("[ERROR] No valid target batteries.")
        return

    print("[INFO] battery id " + str(target_bids))
    os.makedirs(args.out_dir, exist_ok=True)

    for bid in target_bids:
        print("    Inner Adaptation " + str(bid))
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

        pred_mean, pred_std = run_adapt_and_predict(cfg, model, vecizer, meta_thetas, task, max_rul_train)
        if pred_mean is None:
            print("[FAIL] " + str(bid))
            continue
        plot_improved_curve(bid, task, pred_mean, pred_std, args.out_dir)


if __name__ == "__main__":
    main()
