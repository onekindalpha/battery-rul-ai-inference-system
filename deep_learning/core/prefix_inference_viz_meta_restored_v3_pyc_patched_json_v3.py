# -*- coding: utf-8 -*-
"""
prefix_inference_viz_meta.py (restored v3)

Goal: be closer to the original CPython-310 .pyc structure while staying runnable on your repo.
Changes vs prior version:
- Match original top-level function names found in the .pyc: load_cfg_from_ckpt, to_int, safe_to_numpy,
  build_model_and_grouped, make_task_prefix, run_adapt_and_predict, plot_improved_curve, parse_cli_bids, main
- Inline helper logic (dims inference, window cycle alignment, support selection) into the matching functions
  so the code-object inventory resembles the .pyc more closely.
- Use ckpt keys exactly as reported: model_state, meta_thetas, config, seq_scaler, sum_scaler, max_rul_train,
  feature_cols, target_col, train_bids/val_bids/test_bids.

Run:
  python prefix_inference_viz_meta.py --ckpt nasa_bmaml_best_re.pt --eval_dataset from_ckpt --r_ratio 0.5 --out_dir ./viz_meta_test
"""

import os
import argparse
import copy
import json
import datetime
import numpy as np
import torch
from torch.nn.utils.stateless import functional_call

# plotting (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    plt = None

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.models import MultiTaskRULModel
from deep_learning.core.meta_utils import DEVICE, ParamVectorizer, bmaml_inner, make_leaf_thetas
from deep_learning.core.scalers import unscale_rul_array, scale_rul_array


def load_cfg_from_ckpt(cfg_dict):
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def to_int(x):
    
    try:
        return int(x)
    except Exception:
        return x


def safe_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def format_ratio_tag(rr):
    """Return a stable tag like r0p10 for an input ratio 0.10."""
    try:
        rr = float(rr)
    except Exception:
        return "rNA"
    return f"r{rr:.2f}".replace(".", "p")


def parse_cli_bids(bids_str):
    bids_str = (bids_str or "").strip()
    if not bids_str:
        return []
    parts = [p.strip() for p in bids_str.split(",") if p.strip()]
    out = []
    for p in parts:
        s = p.upper()
        if s.startswith("B"):
            s = s[1:]
        out.append(to_int(s))
    return out



def apply_infer_overrides(cfg, inner_steps=None, inner_lr=None):
    """Inference-only overrides without touching the checkpoint cfg."""
    cfg2 = copy.copy(cfg)
    if inner_steps is not None:
        for k in ["inner_steps", "n_inner_steps", "inner_adapt_steps", "inner_iters"]:
            setattr(cfg2, k, int(inner_steps))
    if inner_lr is not None:
        for k in ["inner_lr", "inner_step_size", "inner_alpha", "inner_update_lr"]:
            setattr(cfg2, k, float(inner_lr))
    return cfg2

def build_model_and_grouped(ckpt_path, eval_dataset="from_ckpt"):
    print("[LOAD] Checkpoint: " + str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = load_cfg_from_ckpt(ckpt["config"])

    # lock to training-time feature selection if present
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

    # scalers + max
    seq_scaler = ckpt["seq_scaler"]
    sum_scaler = ckpt["sum_scaler"]
    max_rul_train = float(ckpt["max_rul_train"])

    # model
    state = ckpt["model_state"]

    # infer dims from state (keep inline)
    if getattr(cfg, "use_resnet", False) and ("resnet.stem.weight" in state):
        sd = int(state["resnet.stem.weight"].shape[1])
    else:
        sd = int(state["proj_seq.weight"].shape[1])
    sm = int(state["pm.weight"].shape[1])

    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    vecizer = ParamVectorizer(model)
    meta_thetas = [t.to(DEVICE) for t in ckpt["meta_thetas"]]

    test_bids = ckpt.get("test_bids", sorted(grouped.keys()))
    test_bids = [b for b in test_bids if b in grouped]
    print("[INFO] Eval Batteries (" + str(len(test_bids)) + "): " + str(test_bids))

    return cfg, grouped, model, vecizer, meta_thetas, seq_scaler, sum_scaler, max_rul_train, test_bids


def make_task_prefix(
    bid,
    grouped,
    cfg,
    seq_scaler,
    sum_scaler,
    max_rul_train,
    r_ratio=0.5,
    current_cycle=None,
    min_support=0,
    cap_before_eol=1,
    ratio_base="pos",
):
    g = grouped[bid]

    # window arrays
    seq_w = np.asarray(g["seq"], dtype=np.float32)   # (N_seq, seq_len, F)
    sum_w = np.asarray(g["sum"], dtype=np.float32)   # (N_seq, D)
    rul_raw = np.asarray(g["rul"], dtype=np.float32) # (N_seq,)

    T = int(len(rul_raw))
    if T < 2:
        raise ValueError(f"Not enough windows for bid={bid}: T={T}")

    # align cycles (raw cycles are N_cycle; windows are N_seq)
    cycles_raw = np.asarray(g.get("cycle", np.arange(T, dtype=np.float32)), dtype=np.float32)
    seq_len = int(getattr(cfg, "seq_len", 1))
    start = max(seq_len - 1, 0)
    win_cycles = cycles_raw[start : start + T]
    if len(win_cycles) != T:
        m = min(len(win_cycles), T)
        seq_w, sum_w, rul_raw, win_cycles = seq_w[:m], sum_w[:m], rul_raw[:m], win_cycles[:m]
        T = m

    # EOL index for reporting (last positive RUL + 1)
    pos_idx = np.where(rul_raw > 0)[0]
    eol_idx = int(pos_idx[-1] + 1) if pos_idx.size > 0 else T

    # Cap point near EOL: keeping split strictly BEFORE EOL helps avoid query becoming all-zeros for short-life cells.
    if int(cap_before_eol) == 1 and pos_idx.size > 0:
        eol_cap = max(1, int(eol_idx) - 1)
    else:
        eol_cap = int(eol_idx)

    # decide split
    if current_cycle is not None:
        idx = np.where(win_cycles >= float(current_cycle))[0]
        prefix_len = int(idx[0] + 1) if len(idx) else (T - 1)
        r_ratio_input = None
    else:
        # Decide which length r_ratio refers to:
        # - ratio_base='pos'  : ratio over the positive-RUL region (up to last RUL>0), so r_ratio meaning stays consistent for short-life cells.
        # - ratio_base='full' : ratio over the full window length T (legacy behavior).
        rb = str(ratio_base).lower()
        base_len = int(eol_cap) if (rb == "pos" and pos_idx.size > 0) else int(T)
        prefix_len = int(float(r_ratio) * float(base_len))
        r_ratio_input = float(r_ratio)


    prefix_len = max(1, min(T - 1, prefix_len))
    prefix_len = min(prefix_len, eol_cap)
    # (Inference-only) enforce minimum number of support windows by shifting the split later.
    if int(min_support) > 0:
        prefix_len = max(prefix_len, int(min_support))
        prefix_len = min(prefix_len, eol_cap)

    split_cycle = float(win_cycles[prefix_len - 1])
    eol_ratio = float(prefix_len) / float(eol_idx) if eol_idx > 0 else None

    # support indices (match dataset logic)
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
        # viz (raw)
        "s_cycles_viz": win_cycles[s_idx].astype(np.float32),
        "s_rul_viz": rul_raw[s_idx].astype(np.float32),
        "q_cycles_viz": win_cycles[q_idx].astype(np.float32),
        "q_rul_viz": rul_raw[q_idx].astype(np.float32),
        "split_cycle": split_cycle,
        # info
        "r_ratio_input": r_ratio_input,
        "r_ratio_effective": float(prefix_len) / float(T),
        "eol_ratio": eol_ratio,
        "eol_idx": int(eol_idx),
        "eol_cap": int(eol_cap),
        "current_true_rul": float(rul_raw[prefix_len - 1]),
        "current_cycle_effective": float(win_cycles[prefix_len - 1]),
    }
    return task


def run_adapt_and_predict(cfg, model, vecizer, meta_thetas, task, max_rul_train):

    if os.environ.get("META_DEBUG", "0") == "1":
        try:
            rr = task.get("r_ratio_effective", None)
            print(f"[META_DEBUG] r_ratio_effective={rr} split={task.get('split_cycle', None)} "
                  f"len_s={len(task.get('s_cycles_viz', []))} len_q={len(task.get('q_cycles_viz', []))} "
                  f"s_last={task.get('s_cycles_viz', [None])[-1]} q_first={task.get('q_cycles_viz', [None])[0]}")
        except Exception as e:
            print(f"[META_DEBUG] task_summary failed: {e}")

    # inner adaptation
    dbg = {}
    with torch.enable_grad():
        theta0 = make_leaf_thetas(meta_thetas, detach=True)

        # optional: show "something is happening" (loss/grad) for the loading screen / debug
        if os.environ.get("META_DEBUG", "0") == "1":
            try:
                s_seq = torch.nan_to_num(task["s_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
                s_sum = torch.nan_to_num(task["s_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
                s_rul = torch.nan_to_num(task["s_rul"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

                # first particle only (fast)
                t0 = theta0[0]
                params0 = vecizer.vector_to_params(t0)
                out0 = functional_call(model, params0, (s_seq, s_sum))
                y0 = out0[0] if isinstance(out0, (tuple, list)) else out0
                y0 = torch.nan_to_num(y0.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
                mse0 = ((y0 - s_rul) ** 2).mean()
                g0 = torch.autograd.grad(mse0, t0, retain_graph=False, create_graph=False)[0]
                gnorm0 = float(g0.detach().pow(2).sum().sqrt().cpu().item())

                dbg["support_mse_before_scaled"] = float(mse0.detach().cpu().item())
                dbg["support_grad_norm_before"] = gnorm0
                print(f"[META_DEBUG] support_mse_before_scaled={dbg['support_mse_before_scaled']:.6e} "
                      f"support_grad_norm_before={dbg['support_grad_norm_before']:.6e}")
            except Exception as e:
                print(f"[META_DEBUG] support_loss_pre failed: {e}")

        theta_p, _, eff, *_ = bmaml_inner(
            model,
            vecizer,
            theta0,
            task,
            cfg,
            detach_theta0=True,
            return_losses=False,
        )

        if os.environ.get("META_DEBUG", "0") == "1" and eff and theta_p:
            try:
                # first particle only (fast)
                tp0 = theta_p[0]
                params1 = vecizer.vector_to_params(tp0)
                out1 = functional_call(model, params1, (s_seq, s_sum))
                y1 = out1[0] if isinstance(out1, (tuple, list)) else out1
                y1 = torch.nan_to_num(y1.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
                mse1 = ((y1 - s_rul) ** 2).mean()
                dbg["support_mse_after_scaled"] = float(mse1.detach().cpu().item())
                print(f"[META_DEBUG] support_mse_after_scaled={dbg['support_mse_after_scaled']:.6e}")
            except Exception as e:
                print(f"[META_DEBUG] support_loss_post failed: {e}")


    # DEBUG: check whether inner adaptation actually changed parameters
    if os.environ.get("META_DEBUG", "0") == "1":
        try:
            deltas = []
            for t0, tp in zip(theta0, theta_p):
                d = (tp - t0).pow(2).sum().detach().cpu().item()
                deltas.append(float(d))
            dbg["adapt_delta_sqsum_mean"] = float(np.mean(deltas))
            dbg["adapt_delta_sqsum_min"] = float(np.min(deltas))
            dbg["adapt_delta_sqsum_max"] = float(np.max(deltas))
            print(f"[META_DEBUG] adapt_delta_sqsum mean={dbg['adapt_delta_sqsum_mean']:.6e} min={dbg['adapt_delta_sqsum_min']:.6e} max={dbg['adapt_delta_sqsum_max']:.6e}")
        except Exception as e:
            print(f"[META_DEBUG] adapt_delta_sqsum failed: {e}")

    task["debug"] = dbg

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
            preds.append(safe_to_numpy(y_hat.squeeze(-1)))

    if len(preds) == 0:
        return None, None

    preds_stack = np.stack(preds, axis=0)
    mean_sc = preds_stack.mean(axis=0)
    std_sc = preds_stack.std(axis=0)

    pred_mean = unscale_rul_array(mean_sc, getattr(cfg, "rul_mode", "minmax"), max_rul_train)

    # std in real units via +1σ bound (then plot ±2σ)
    upper_sc = mean_sc + std_sc
    upper_real = unscale_rul_array(upper_sc, getattr(cfg, "rul_mode", "minmax"), max_rul_train)
    pred_std = np.abs(upper_real - pred_mean)

    return np.asarray(pred_mean, dtype=float), np.asarray(pred_std, dtype=float)


def plot_improved_curve(bid, task, pred_mean, pred_std, out_dir=None):
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

    title = f"{bid} | BMAML-SVGD | RMSE={rmse:.3f}, MAE={mae:.3f} | len_s={len(s_true)}, len_q={len(q_true)}"
    if task.get("eol_ratio", None) is not None:
        title += f" (End of Life) NrR={float(task['eol_ratio']):.2f}"
    if task.get("r_ratio_input", None) is not None:
        title += f" (r_in={float(task['r_ratio_input']):.2f})"
    if task.get("r_ratio_effective", None) is not None:
        title += f" (r_eff={float(task['r_ratio_effective']):.2f})"

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

    if out_dir is None:
        out_dir = "./viz_meta_test"
    os.makedirs(out_dir, exist_ok=True)
    rr_tag = task.get("r_ratio_tag", None)
    tag = format_ratio_tag(rr_tag) if rr_tag is not None else format_ratio_tag(task.get("r_ratio_input", None))
    path = os.path.join(out_dir, f"{bid}_viz_meta_{tag}.png")
    plt.savefig(path)
    print("[PLOT] Saved to " + str(path))
    plt.close()


def save_viz_json(bid, task, pred_mean, pred_std, out_dir, tag):
    """Save per-battery inference result as JSON for web precompute."""
    s_cyc = np.asarray(task.get("s_cycles_viz", []), dtype=float)
    s_true = np.asarray(task.get("s_rul_viz", []), dtype=float)
    q_cyc = np.asarray(task.get("q_cycles_viz", []), dtype=float)
    q_true = np.asarray(task.get("q_rul_viz", []), dtype=float)

    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)

    m = int(min(len(q_cyc), len(pred_mean)))
    q_cyc = q_cyc[:m]
    q_true = q_true[:m]
    pred_mean = pred_mean[:m]
    pred_std = pred_std[:m]

    rmse = float(np.sqrt(np.mean((pred_mean - q_true) ** 2))) if m else float("nan")
    mae = float(np.mean(np.abs(pred_mean - q_true))) if m else float("nan")

    payload = {
        "battery_id": str(bid),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "r_ratio_input": task.get("r_ratio_input", None),
        "r_ratio_effective": task.get("r_ratio_effective", None),
        "eol_ratio": task.get("eol_ratio", None),
        "split_cycle": task.get("split_cycle", None),
        "len_s": int(len(s_cyc)),
        "len_q": int(len(q_cyc)),
        "q_pos": int((q_true > 0).sum()) if len(q_true) else 0,
        "current_true_rul": task.get("current_true_rul", None),
        "current_cycle_effective": task.get("current_cycle_effective", None),
        "metrics": {"rmse": rmse, "mae": mae},
        "support": {"cycle": s_cyc.tolist(), "rul": s_true.tolist()},
        "query": {"cycle": q_cyc.tolist(), "true_rul": q_true.tolist()},
        "pred": {"mean": pred_mean.tolist(), "std": pred_std.tolist()},
        "debug": task.get("debug", None),
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{bid}_viz_meta_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("[JSON] Saved to " + str(path))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--r_ratio", type=float, default=0.5)
    parser.add_argument("--current_cycle", type=float, default=None)
    parser.add_argument("--eval_dataset", type=str, default="from_ckpt", choices=["from_ckpt", "nasa", "cacle", "both"])
    parser.add_argument("--bids", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./viz_meta_test")
    parser.add_argument("--save_json", type=int, default=1, help="1: save per-battery JSON alongside PNG")
    parser.add_argument("--save_batch_json", type=int, default=1, help="1: save one combined JSON for all target batteries")
    parser.add_argument("--min_support", type=int, default=0, help="Inference-only: enforce at least this many support windows by shifting split later")
    parser.add_argument("--cap_before_eol", type=int, default=1, help="1: keep split strictly before EOL (prevents query becoming all-zeros for short-life cells)")
    parser.add_argument("--ratio_base", type=str, default="pos", choices=["pos","full"], help="How to interpret r_ratio: pos=ratio over positive-RUL region (up to last RUL>0), full=ratio over full window length T (legacy)")
    parser.add_argument("--infer_inner_steps", type=int, default=None, help="Inference-only override for inner adaptation steps (if bmaml_inner reads cfg.inner_steps)")
    parser.add_argument("--infer_inner_lr", type=float, default=None, help="Inference-only override for inner adaptation lr (if bmaml_inner reads cfg.inner_lr)")

    args = parser.parse_args()

    cfg, grouped, model, vecizer, meta_thetas, seq_scaler, sum_scaler, max_rul_train, test_bids = build_model_and_grouped(
        args.ckpt, args.eval_dataset
    )

    cfg_infer = apply_infer_overrides(cfg, args.infer_inner_steps, args.infer_inner_lr)

    # map numeric id -> grouped key
    id_to_key = {}
    for k in grouped.keys():
        digits = "".join(ch for ch in str(k) if ch.isdigit())
        if digits:
            try:
                id_to_key[int(digits)] = k
            except Exception:
                pass

    if args.bids:
        req = parse_cli_bids(args.bids)
        target_bids = [id_to_key[x] for x in req if x in id_to_key]
    else:
        target_bids = test_bids

    if not target_bids:
        print("[ERROR] No valid target batteries.")
        return

    print("[INFO] Using target batteries: " + str(target_bids))
    os.makedirs(args.out_dir, exist_ok=True)
    tag = format_ratio_tag(args.r_ratio) if args.current_cycle is None else format_ratio_tag("NA")
    batch_payloads = []

    for bid in target_bids:
        print("    Inner Adaptation " + str(bid))
        task = make_task_prefix(
            bid,
            grouped,
            cfg_infer,
            seq_scaler,
            sum_scaler,
            max_rul_train,
            r_ratio=args.r_ratio,
            current_cycle=args.current_cycle,
            min_support=args.min_support,
            cap_before_eol=args.cap_before_eol,
            ratio_base=args.ratio_base,
        )
        task["r_ratio_tag"] = float(args.r_ratio) if args.current_cycle is None else task.get("r_ratio_effective", None)
        pred_mean, pred_std = run_adapt_and_predict(cfg_infer, model, vecizer, meta_thetas, task, max_rul_train)
        if pred_mean is None:
            print("[FAIL] " + str(bid))
            continue
        plot_improved_curve(bid, task, pred_mean, pred_std, args.out_dir)
        if int(getattr(args, "save_json", 1)) == 1:
            payload = save_viz_json(bid, task, pred_mean, pred_std, args.out_dir, tag)
            batch_payloads.append(payload)
    if int(getattr(args, "save_batch_json", 1)) == 1 and len(batch_payloads) > 0:
        batch_path = os.path.join(args.out_dir, f"batch_viz_meta_{tag}.json")
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump({"tag": tag, "created_at": datetime.datetime.now().isoformat(timespec="seconds"), "items": batch_payloads}, f, ensure_ascii=False)
        print("[JSON] Saved batch to " + str(batch_path))



if __name__ == "__main__":
    main()
