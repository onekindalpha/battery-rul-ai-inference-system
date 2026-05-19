import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CSV 로드
# ============================================================
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned.csv')
print(f"🔄 로드 완료: {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# 1️⃣ 용량 관련 피처 (4개)
# ============================================================
print("\n📌 [1/6] 용량 관련 피처 생성...")

df['soh'] = df['capacity_ahr'] / df['capacity_after_regen']
df['capacity_norm'] = df['capacity_ahr'] / df['capacity_after_regen']
df['capacity_derivative'] = df.groupby('battery')['capacity_ahr'].diff()
df['regen_strength'] = df['capacity_after_regen'] - df['capacity_ahr']

# ============================================================
# 2️⃣ 임피던스 관련 (3개)
# ============================================================
print("📌 [2/6] 임피던스 관련 피처 생성...")

df['impedance_sum'] = df['re_ohm_interp'] + df['rct_ohm_interp']

initial_impedance = df.groupby('battery')['impedance_sum'].transform('first')
df['impedance_growth'] = ((df['impedance_sum'] - initial_impedance) / (initial_impedance + 1e-9) * 100).fillna(0)

initial_dcr = df.groupby('battery')['dcr'].transform('first')
df['dcr_growth'] = ((df['dcr'] - initial_dcr) / (initial_dcr + 1e-9) * 100).fillna(0)

# ============================================================
# 3️⃣ 열화 지표 스케일링 & 평활 (6개)
# ============================================================
print("📌 [3/6] 열화 지표 스케일링 및 평활...")

for col in ['lli', 'lam', 'dcr']:
    mean = df[col].mean()
    std = df[col].std()
    df[f'{col}_scaled'] = np.clip((df[col] - mean) / std, -3, 3)

    df[f'{col}_smooth'] = df.groupby('battery')[col].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )

# ============================================================
# 4️⃣ 스트레스 / 열 관련 (8개)
# ============================================================
print("📌 [4/6] 스트레스/열 관련 피처 생성...")

df['temp_rise'] = df['temperature_measured_max'] - df['ambient_temp_c']
df['thermal_stress'] = df['temp_rise'] / (df['discharge_time_sec'] + 1e-9)

df['ir_drop'] = df['voltage_load_mean'] - df['voltage_measured_mean']
df['voltage_range'] = df['voltage_measured_max'] - df['voltage_measured_min']

df['C_rate_avg'] = df['current_measured_mean'] / (df['capacity_after_regen'] + 1e-9)
df['C_rate_max'] = (df['current_measured_mean'] + 2 * df['current_measured_std']) / (df['capacity_after_regen'] + 1e-9)

df['current_temp_product'] = df['current_measured_mean'] * df['temp_rise']
df['load_temp_interact'] = df['C_rate_avg'] * df['temp_rise']

# ============================================================
# 5️⃣ 시간/사이클 파생 (2개)
# ============================================================
print("📌 [5/6] 시간/사이클 파생 피처 생성...")

df['cycle_diff_time'] = df.groupby('battery')['time_seconds'].diff()
df['cycle_duration_rate'] = df['cycle_diff_time'] / (df['discharge_time_sec'] + 1e-9)

# ============================================================
# 6️⃣ 인코딩 & 구간화 (3개)
# ============================================================
print("📌 [6/6] 인코딩 및 구간화 생성...")

le = LabelEncoder()
df['battery_encoded'] = le.fit_transform(df['battery'])

df['_crate_bin'] = pd.cut(df['C_rate_avg'], bins=5, labels=False, duplicates='drop')
df['_temp_bin'] = pd.cut(df['temp_rise'], bins=5, labels=False, duplicates='drop')

# ============================================================
# 최종 저장
# ============================================================
print(f"\n✅ 완료! 최종 컬럼 수: {len(df.columns)}")
df.to_csv('battery_training_data_completed.csv', index=False)
print(f"💾 저장: battery_training_data_completed.csv")