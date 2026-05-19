"""BMAML runtime helpers for FastAPI backend.

This module wraps the existing prefix_inference_viz_meta utilities and
exposes a simple interface:

- load_meta_state(ckpt_path, eval_dataset)
- build_runtime_records(meta_state, r_ratio) -> Dict[battery_id, record_dict]

Each record_dict contains:
    - battery_id
    - cycles: List[float]
    - rul_true: List[float]
    - rul_pred: List[float]
    - rul_std: List[float]
    - split_cycle: float
    - rmse: float
    - mae: float
    - cap_init: float
    - cap_final: float
    - cycle_life_obs: float
    - capacity_curve: Optional[List[float]]
"""

from typing import Dict

import numpy as np

from deep_learning.core.prefix_inference_viz_meta_restored_v3_pyc import (
    build_model_and_grouped,
    make_task_prefix,
    run_adapt_and_predict,
)


def load_meta_state(ckpt_path: str, eval_dataset: str = "from_ckpt"):
    """Load BMAML meta-state from checkpoint.

    Returns the full tuple:
    (cfg, grouped, model, vecizer, meta_thetas, seq_scaler, sum_scaler, max_rul_train, test_bids)
    """
    return build_model_and_grouped(ckpt_path, eval_dataset=eval_dataset)


def build_runtime_records(meta_state, r_ratio: float = 0.3) -> Dict[str, dict]:
    """Run BMAML for all test_bids and build dashboard-ready records.

    This mirrors the dashboard's "실시간 추론"용 전처리 로직 (지원/질의 분리 + full curve).
    """
    (
        cfg,
        grouped,
        model,
        vecizer,
        meta_thetas,
        seq_scaler,
        sum_scaler,
        max_rul_train,
        test_bids,
    ) = meta_state

    records: Dict[str, dict] = {}

    for bid in test_bids:
        if bid not in grouped:
            continue

        g = grouped[bid]

        # Build a single prefix-style task for this battery
        task = make_task_prefix(
            bid=bid,
            grouped=grouped,
            cfg=cfg,
            seq_scaler=seq_scaler,
            sum_scaler=sum_scaler,
            max_rul_train=max_rul_train,
            r_ratio=r_ratio,
            current_cycle=None,
        )

        pred_mean, pred_std = run_adapt_and_predict(
            cfg=cfg,
            model=model,
            vecizer=vecizer,
            meta_thetas=meta_thetas,
            task=task,
            max_rul_train=max_rul_train,
        )

        # Support / query decomposition from task
        s_cyc = np.asarray(task["s_cycles_viz"], dtype=float)
        s_true = np.asarray(task["s_rul_viz"], dtype=float)
        q_cyc = np.asarray(task["q_cycles_viz"], dtype=float)
        q_true = np.asarray(task["q_rul_viz"], dtype=float)
        split_cycle = float(task["split_cycle"])

        pred_mean = np.asarray(pred_mean, dtype=float)
        pred_std = np.asarray(pred_std, dtype=float)

        # Build full curves: history (support) + prediction (query)
        cycles_full = np.concatenate([s_cyc, q_cyc])
        rul_true_full = np.concatenate([s_true, q_true])

        hist_nan = np.full_like(s_true, np.nan, dtype=float)
        pred_full = np.concatenate([hist_nan, pred_mean])
        std_full = np.concatenate([hist_nan, pred_std])

        # Metrics on query region only
        if q_true.size > 0:
            diff = pred_mean - q_true
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
        else:
            rmse = float("nan")
            mae = float("nan")

        # Capacity curve & observed life
        cap_curve = g.get("capacity_curve", None)
        if cap_curve is not None and len(cap_curve) > 0:
            cap_arr = np.asarray(cap_curve, dtype=float)
            cap_init = float(cap_arr[0])
            cap_final = float(cap_arr[-1])
            cap_list = cap_arr.tolist()
        else:
            cap_init = float("nan")
            cap_final = float("nan")
            cap_list = None

        cyc_arr = np.asarray(g.get("cycle", []), dtype=float)
        cycle_life_obs = float(cyc_arr.max()) if cyc_arr.size > 0 else float("nan")

        rec = {
            "battery_id": bid,
            "cycles": cycles_full.tolist(),
            "rul_true": rul_true_full.tolist(),
            "rul_pred": pred_full.tolist(),
            "rul_std": std_full.tolist(),
            "split_cycle": split_cycle,
            "rmse": rmse,
            "mae": mae,
            "cap_init": cap_init,
            "cap_final": cap_final,
            "cycle_life_obs": cycle_life_obs,
            "capacity_curve": cap_list,
        }

        # Use string key so API/JSON side is consistent
        records[str(bid)] = rec

    return records
