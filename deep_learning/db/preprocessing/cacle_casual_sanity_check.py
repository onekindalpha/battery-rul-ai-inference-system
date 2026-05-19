import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/db/cacle_battery_training_data_cleaned_final_causal.csv"
df = pd.read_csv(path)

# 예시: CS2_35 하나만 보기
bid = "CS2_38"
g = df[df["battery"] == bid].sort_values("cycle_num")

print("battery:", bid)
print("min SoH:", g["soh"].min(), "max SoH:", g["soh"].max())
print("unique cycle_life:", g["cycle_life"].dropna().unique())

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axes[0].plot(g["cycle_num"], g["capacity_ahr"])
axes[0].set_ylabel("Capacity (Ah)")

axes[1].plot(g["cycle_num"], g["soh"])
axes[1].axhline(0.8, linestyle="--")
axes[1].set_ylabel("SoH")

axes[2].plot(g["cycle_num"], g["rul_cycles"])
axes[2].set_ylabel("RUL (cycles)")
axes[2].set_xlabel("Cycle")

plt.tight_layout()
plt.show()
