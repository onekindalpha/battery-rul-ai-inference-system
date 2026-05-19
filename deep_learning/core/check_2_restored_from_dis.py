# -*- coding: utf-8 -*-
"""
Restored from dis dump: check_2.dis.txt

This is a small diagnostic script that:
- imports build_model_and_grouped from deep_learning.core.prefix_inference_viz_meta
- loads a BMAML checkpoint path hard-coded in the original
- prints test_bids, inspects grouped[test_bids[0]] keys if possible
"""

from deep_learning.core.prefix_inference_viz_meta_restored_v3_pyc import build_model_and_grouped

ckpt_path = "/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/core_checkpoints/nasa_bmaml_best_re.pt"

meta_state = build_model_and_grouped(ckpt_path, eval_dataset="from_ckpt")
cfg, grouped, model, vecizer, meta_thetas, seq_scaler, sum_scaler, max_rul_train, test_bids = meta_state

print("test_bids:", test_bids)

g = grouped[test_bids[0]]

print(type(g))

try:
    print("g keys:", g.keys())
except Exception as e:
    print("no .keys(), dir(g):", dir(g))

print("all_grouped_bids:", sorted(grouped.keys()))
print("train_candidates(=all - test):", sorted(set(grouped.keys()) - set(test_bids)))

# cfg 안에 train/val/eval/test 관련 필드가 저장돼 있으면 같이 출력
d = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", {})
keys = [k for k in d.keys() if any(t in k.lower() for t in ["train", "val", "eval", "test", "bid", "split", "fold"])]
print("cfg_split_keys:", keys)
for k in keys:
    print(f"{k}:", d[k])
