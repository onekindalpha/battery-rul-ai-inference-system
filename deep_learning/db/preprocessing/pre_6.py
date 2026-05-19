"""
EOL_DCYCLE 계산 분석 스크립트
각 배터리가 threshold 이하로 내려가는 과정을 상세히 보여줍니다.
"""

import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned.csv')
eol_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_eol_info.csv')

print("="*50)
print("EOL_DCYCLE 계산 로직")
print("="*50)
print("\n정의:")
print("  1. eol_threshold = capacity_after_regen * 0.5  (50% 감소 지점)")
print("  2. eol_dcycle = capacity_ahr이 threshold 이하로 내려가는 첫 번째 cycle\n")

# ============================================================================
# 예시 배터리 1: B0005 (정상적인 감소)
# ============================================================================
print("\n" + "="*50)
print("예시 1: B0005 (정상적인 감소 곡선)")
print("="*50)

bid = 'B0005'
batt_data = df[df['battery'] == bid].sort_values('cycle_num')
batt_eol = eol_df[eol_df['battery'] == bid].iloc[0]

capacity_after_regen = batt_eol['capacity_after_regen']
eol_threshold = batt_eol['eol_threshold']
eol_dcycle = int(batt_eol['eol_dcycle'])

print(f"\n🔍 기본 정보:")
print(f"  배터리: {bid}")
print(f"  최대 용량 (capacity_after_regen): {capacity_after_regen:.4f} Ahr")
print(f"  EOL threshold (70%): {eol_threshold:.4f} Ahr")
print(f"  계산: {capacity_after_regen:.4f} × 0.7 = {eol_threshold:.4f}\n")

print(f"📍 EOL 포인트: cycle {eol_dcycle}\n")

# EOL 주변 데이터 보기
print("📊 용량 변화 (cycle 155~168):")
print(f"{'Cycle':>6} | {'Capacity':>10} | {'Status':>20}")
print("-" * 40)

view_range = batt_data[(batt_data['cycle_num'] >= 155) & (batt_data['cycle_num'] <= 168)]
for _, row in view_range.iterrows():
    cycle = int(row['cycle_num'])
    capacity = row['capacity_ahr']

    if cycle == eol_dcycle:
        status = "⬅️ EOL 지점 (≤ threshold)"
        marker = ">>> "
    elif capacity <= eol_threshold:
        status = "✓ threshold 이하"
        marker = "    "
    else:
        status = "○ threshold 초과"
        marker = "    "

    print(f"{marker}{cycle:>3d} | {capacity:>10.4f} | {status}")

# ============================================================================
# 예시 배터리 2: B0006 (중간 속도 감소)
# ============================================================================
print("\n" + "="*80)
print("예시 2: B0006 (중간 속도의 감소 곡선)")
print("="*80)

bid = 'B0006'
batt_data = df[df['battery'] == bid].sort_values('cycle_num')
batt_eol = eol_df[eol_df['battery'] == bid].iloc[0]

capacity_after_regen = batt_eol['capacity_after_regen']
eol_threshold = batt_eol['eol_threshold']
eol_dcycle = int(batt_eol['eol_dcycle'])

print(f"\n🔍 기본 정보:")
print(f"  배터리: {bid}")
print(f"  최대 용량: {capacity_after_regen:.4f} Ahr")
print(f"  EOL threshold: {eol_threshold:.4f} Ahr")

print(f"\n📍 EOL 포인트: cycle {eol_dcycle}\n")

print("📊 용량 변화 (cycle 95~110):")
print(f"{'Cycle':>6} | {'Capacity':>10} | {'Status':>20}")
print("-" * 40)

view_range = batt_data[(batt_data['cycle_num'] >= 95) & (batt_data['cycle_num'] <= 110)]
for _, row in view_range.iterrows():
    cycle = int(row['cycle_num'])
    capacity = row['capacity_ahr']

    if cycle == eol_dcycle:
        status = "⬅️ EOL 지점"
        marker = ">>> "
    elif capacity <= eol_threshold:
        status = "✓ threshold 이하"
        marker = "    "
    else:
        status = "○ threshold 초과"
        marker = "    "

    print(f"{marker}{cycle:>3d} | {capacity:>10.4f} | {status}")

# ============================================================================
# 예시 배터리 3: B0033 (초반 급락)
# ============================================================================
print("\n" + "="*80)
print("예시 3: B0033 (초반 급락 - 불량 배터리)")
print("="*80)

bid = 'B0033'
batt_data = df[df['battery'] == bid].sort_values('cycle_num')
batt_eol = eol_df[eol_df['battery'] == bid].iloc[0]

capacity_after_regen = batt_eol['capacity_after_regen']
eol_threshold = batt_eol['eol_threshold']
eol_dcycle = int(batt_eol['eol_dcycle'])

print(f"\n🔍 기본 정보:")
print(f"  배터리: {bid}")
print(f"  최대 용량: {capacity_after_regen:.4f} Ahr")
print(f"  EOL threshold: {eol_threshold:.4f} Ahr")
print(f"  ⚠️  주목: cycle 1부터 threshold 이하!")

print(f"\n📍 EOL 포인트: cycle {eol_dcycle}\n")

print("📊 용량 변화 (cycle 1~10):")
print(f"{'Cycle':>6} | {'Capacity':>10} | {'Status':>20}")
print("-" * 40)

view_range = batt_data[(batt_data['cycle_num'] >= 1) & (batt_data['cycle_num'] <= 10)]
for _, row in view_range.iterrows():
    cycle = int(row['cycle_num'])
    capacity = row['capacity_ahr']

    if cycle == eol_dcycle:
        status = "⬅️ EOL 지점"
        marker = ">>> "
    elif capacity <= eol_threshold:
        status = "✓ threshold 이하"
        marker = "    "
    else:
        status = "○ threshold 초과"
        marker = "    "

    print(f"{marker}{cycle:>3d} | {capacity:>10.4f} | {status}")

# ============================================================================
# 전체 요약
# ============================================================================
print("\n" + "="*80)
print("전체 배터리 EOL 통계")
print("="*80 + "\n")

print("EO L_DCYCLE 분포:")
print(f"  최소: {eol_df['eol_dcycle'].min():.0f} cycles")
print(f"  최대: {eol_df['eol_dcycle'].max():.0f} cycles")
print(f"  평균: {eol_df['eol_dcycle'].mean():.1f} cycles")
print(f"  중앙값: {eol_df['eol_dcycle'].median():.1f} cycles\n")

# EOL 범위별 그룹화
print("EOL_DCYCLE 범위별 배터리 수:")
early_eol = eol_df[eol_df['eol_dcycle'] <= 6]
mid_eol = eol_df[(eol_df['eol_dcycle'] > 6) & (eol_df['eol_dcycle'] <= 50)]
late_eol = eol_df[eol_df['eol_dcycle'] > 50]

print(f"  1~6 cycles (조기 실패):     {len(early_eol):>2d}개 - {', '.join(early_eol['battery'].tolist())}")
print(f"  7~50 cycles (중기 실패):     {len(mid_eol):>2d}개 - {', '.join(mid_eol['battery'].tolist())}")
print(f"  50+ cycles (정상 작동):      {len(late_eol):>2d}개 - {', '.join(late_eol['battery'].tolist())}")

print("\n" + "="*80)
print("전체 배터리 데이터:")
print("="*80 + "\n")

display_cols = ['battery', 'capacity_after_regen', 'eol_threshold', 'eol_dcycle', 'cluster']
print(eol_df[display_cols].to_string(index=False))

print("\n\n✅ 계산 완료!")