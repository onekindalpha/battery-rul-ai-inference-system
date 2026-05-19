import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ 데이터 로드
# -----------------------------
df = pd.read_csv("/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/db/battery_training_data_cleaned_final.csv")

# 배터리 ID 및 cycle 정렬
df = df.rename(columns={'battery': 'battery_id', 'cycle_num': 'cycle'}, errors='ignore')
df = df.sort_values(['battery_id', 'cycle'])

# -----------------------------
# 2️⃣ 용량 단조화(Smoothing) 적용
# -----------------------------
# (각 배터리별로 capacity가 오직 감소하도록 누적 최소 적용)
df['capacity_smooth'] = df.groupby('battery_id')['capacity_ahr'].cummin()

# -----------------------------
# 3️⃣ 회복 셀(비정상) 탐지 및 필터링
# -----------------------------
increase_ratios = []
for bid, grp in df.groupby('battery_id'):
    cap = grp['capacity_ahr'].values
    if len(cap) < 5:
        continue
    inc_ratio = np.sum(np.diff(cap) > 0) / len(cap)
    increase_ratios.append((bid, inc_ratio))

inc_df = pd.DataFrame(increase_ratios, columns=['battery_id', 'increase_ratio'])
bad_batteries = inc_df.loc[inc_df['increase_ratio'] > 0.2, 'battery_id'].tolist()
print(f"⚠️ 회복 셀 제거 대상 {len(bad_batteries)}개:", bad_batteries)

# 회복 셀 제외
df_filtered = df[~df['battery_id'].isin(bad_batteries)].copy()

# -----------------------------
# 4️⃣ EOL 재계산 (80% 용량 기준)
# -----------------------------
eol_info = {}
for bid, grp in df_filtered.groupby('battery_id'):
    cap0 = grp['capacity_smooth'].iloc[0]
    eol_cap = 0.8 * cap0
    eol_cycle = grp.loc[grp['capacity_smooth'] <= eol_cap, 'cycle']
    eol_info[bid] = eol_cycle.min() if len(eol_cycle) > 0 else grp['cycle'].max()

eol_df = pd.DataFrame.from_dict(eol_info, orient='index', columns=['eol_cycle'])
print("\n🔹 EOL 정보 (상위 5개):")
print(eol_df.head())

# -----------------------------
# 5️⃣ 시각화 (보정 전/후 비교)
# -----------------------------
sample_bids = np.random.choice(df_filtered['battery_id'].unique(), 4, replace=False)

plt.figure(figsize=(10, 6))
for bid in sample_bids:
    grp = df[df['battery_id'] == bid]
    plt.plot(grp['cycle'], grp['capacity_ahr'], alpha=0.3, label=f"{bid} (raw)")
    plt.plot(grp['cycle'], grp['capacity_smooth'], lw=2, label=f"{bid} (smoothed)")
plt.xlabel("Cycle")
plt.ylabel("Capacity (A·h)")
plt.title("용량 단조화 전후 비교")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6️⃣ 결과 저장
# -----------------------------
df_filtered.to_csv("/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/db/battery_training_data_cleaned_smooth_filtered.csv", index=False)
print("\n✅ 보정 완료: 'battery_training_data_cleaned_smooth_filtered.csv' 로 저장됨")
