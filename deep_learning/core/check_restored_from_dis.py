import torch

ckpt_path = "/Users/velocitygoal/Desktop/battery_project/v11/core_checkpoints/nasa_bmaml_best_re.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("TRAIN:", ckpt.get("train_bids"))
print("VAL  :", ckpt.get("val_bids"))
print("TEST :", ckpt.get("test_bids"))
print("CONFIG dataset_source:", ckpt["config"].get("dataset_source"))
