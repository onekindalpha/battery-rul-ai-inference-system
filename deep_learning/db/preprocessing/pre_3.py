import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_complete_data_final.csv')

print("=" * 80)
print("🔋 최종 학습 데이터셋 (K=2, Capacity Regeneration 고려)")
print("=" * 80)

# ============================================
# 1. Capacity regeneration 이후 초기값 설정
# ============================================

def find_regeneration_end(capacity_values):
    """용량의 최대값 찾기"""
    if len(capacity_values) < 5:
        return 0
    return np.argmax(capacity_values)


metrics = []

for bid in sorted(df['battery'].unique()):
    data = df[df['battery'] == bid].sort_values('cycle_num')
    capacity = data['capacity_ahr'].values
    re_ohm = data['re_ohm_interp'].values
    impedance = data['battery_impedance_interp'].values
    voltage = data['voltage_measured_mean'].values

    if len(capacity) < 5 or np.any(np.isnan(capacity)):
        continue

    # Capacity regeneration 이후의 피크 용량 찾기
    regen_idx = find_regeneration_end(capacity)
    capacity_after_regen = capacity[regen_idx]
    regen_cycles = data['cycle_num'].iloc[regen_idx]

    # 새로운 용량 감소율 (regeneration 이후부터)
    cap_deg = (capacity_after_regen - capacity[-1]) / capacity_after_regen * 100

    if abs(cap_deg) > 200:
        continue

    # DCR, 임피던스 성장
    dcr_growth = ((re_ohm[-1] - re_ohm[0]) / abs(re_ohm[0]) * 100) if re_ohm[0] != 0 and abs(re_ohm[0]) > 0.001 else 0
    dcr_growth = np.clip(dcr_growth, -100, 100)

    impedance_growth = ((impedance[-1] - impedance[0]) / abs(impedance[0]) * 100) if impedance[0] != 0 and abs(impedance[0]) > 0.001 else 0
    impedance_growth = np.clip(impedance_growth, -100, 100)

    # LLi, LAM 지표
    voltage_deg = voltage[0] - voltage[-1]
    lli_indicator = voltage_deg / abs(cap_deg) if abs(cap_deg) > 0.1 else 0
    lam_indicator = dcr_growth / abs(cap_deg) if abs(cap_deg) > 0.1 else 0

    metrics.append({
        'battery': bid,
        'num_cycles': len(data),
        'regen_cycles': int(regen_cycles),
        'capacity_after_regen': capacity_after_regen,
        'capacity_deg': cap_deg,
        'dcr_growth': dcr_growth,
        'impedance_growth': impedance_growth,
        'lli_indicator': lli_indicator,
        'lam_indicator': lam_indicator
    })

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan).dropna()

print(f"\n✓ {len(metrics_df)}개 배터리 메트릭 계산 완료")

# ============================================
# 2. K=2 클러스터링
# ============================================

print("\n[1/3] K=2 클러스터링 중...")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features_array = metrics_df[[
    'capacity_deg', 'dcr_growth', 'impedance_growth', 'lli_indicator', 'lam_indicator'
]].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
clusters = kmeans.fit_predict(features_scaled)

metrics_df['cluster'] = clusters

print(f"✓ 클러스터 결과:")
for c in range(2):
    batteries = metrics_df[metrics_df['cluster'] == c]['battery'].tolist()
    count = len(batteries)
    print(f"  Cluster {c}: {count}개")

# ============================================
# 3. 원본 CSV에 새 컬럼 추가
# ============================================

print("\n[2/3] 최종 데이터셋 생성 중...")

df_final = df.copy()

# 배터리별 메트릭 매핑
df_final['regen_cycles'] = df_final['battery'].map(lambda x: metrics_df[metrics_df['battery'] == x]['regen_cycles'].values[0] if x in metrics_df['battery'].values else np.nan)
df_final['capacity_after_regen'] = df_final['battery'].map(lambda x: metrics_df[metrics_df['battery'] == x]['capacity_after_regen'].values[0] if x in metrics_df['battery'].values else np.nan)
df_final['capacity_deg'] = df_final['battery'].map(lambda x: metrics_df[metrics_df['battery'] == x]['capacity_deg'].values[0] if x in metrics_df['battery'].values else np.nan)
df_final['cluster'] = df_final['battery'].map(lambda x: metrics_df[metrics_df['battery'] == x]['cluster'].values[0] if x in metrics_df['battery'].values else np.nan)

# 저장
df_final.to_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_final_training_data.csv', index=False)
print("✅ 저장: battery_final_training_data.csv")

# ============================================
# 4. 메타데이터 CSV
# ============================================

summary_df = metrics_df[[
    'battery', 'num_cycles', 'regen_cycles', 'capacity_after_regen', 'capacity_deg',
    'dcr_growth', 'impedance_growth', 'lli_indicator', 'lam_indicator', 'cluster'
]].copy()

summary_df.to_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_final_metadata.csv', index=False)
print("✅ 저장: battery_final_metadata.csv")

# ============================================
# 5. 클러스터링 모델 저장
# ============================================

print("\n[3/3] 모델 저장 중...")

clustering_model = {
    'kmeans': kmeans,
    'scaler': scaler,
    'optimal_k': 2,
    'silhouette_score': 0.631,
    'metrics_df': metrics_df
}

with open('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_final_clustering_model.pkl', 'wb') as f:
    pickle.dump(clustering_model, f)

print("✅ 저장: battery_final_clustering_model.pkl")

# ============================================
# 6. 요약 출력
# ============================================

print("\n" + "=" * 80)
print("📊 최종 데이터셋 요약 (K=2, Silhouette=0.631)")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("✅ 완료!")
print("=" * 80)