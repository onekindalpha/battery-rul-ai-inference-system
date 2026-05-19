import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 기준 경로 (필요하면 절대경로로 바꿔서 써도 됨)
ROOT_DIR = Path(__file__).resolve().parents[2]  # deep_learning/db/preprocessing/... 기준
DB_DIR = ROOT_DIR / "db"
CAUSAL_DB = DB_DIR / "battery_training_data_cleaned_final_causal.csv"

df = pd.read_csv(CAUSAL_DB)

# 핵심 임피던스 관련 피처 NaN 비율
print(df[["re_ohm_interp", "dcr", "impedance_sum", "lam"]].isna().mean())
print(df.isna().sum().sort_values(ascending=False).head(10))

# Re proxy fallback(1e-3) 사용 비율
print("Re==1e-3 ratio:", (np.isclose(df["re_ohm_interp"], 1e-3)).mean())

# 배터리별 fallback 비율
print(
    df.groupby("battery")["re_ohm_interp"]
      .apply(lambda x: np.isclose(x, 1e-3).mean())
      .describe()
)

# 극단치 / 스케일 체크
cols = [
    "capacity_ahr",
    "soh",
    "re_ohm_interp",
    "dcr_growth",
    "impedance_growth",
    "lli",
    "lam",
    "thermal_stress",
]
print(df[cols].describe(percentiles=[.01, .05, .5, .95, .99]))

# 상수/거의 상수 컬럼 체크
stds = df.select_dtypes(np.number).std()
print(stds.sort_values().head(10))  # std 거의 0인 애들

# 배터리별 길이/분포 체크
lens = df.groupby("battery")["cycle_num"].max()
print(lens.describe())
print(lens.sort_values().head())

# RUL / cycle_life 분포 확인
print(df[["cycle_life", "rul_cycles", "rul_norm"]].describe())
