"""
2단계: EOL 정보 생성
========================================
정제된 데이터에서 배터리별 EOL 정보 계산

입력: battery_training_data_cleaned.csv
출력: battery_eol_info.csv

계산 로직:
- capacity_after_regen: capacity_ahr의 최대값
- eol_threshold: capacity_after_regen × 0.7 (30% 감소)
- eol_dcycle: capacity_ahr이 threshold 이하로 떨어지는 첫 번째 cycle
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_eol_info(csv_path):
    """
    배터리 데이터에서 EOL 정보 계산

    Parameters:
    -----------
    csv_path : str
        배터리 데이터 CSV 파일 경로 (battery_training_data_cleaned.csv)

    Returns:
    --------
    pd.DataFrame
        배터리별 EOL 정보 데이터프레임
    """

    print("="*80)
    print("2단계: EOL 정보 생성")
    print("="*80 + "\n")

    # 파일 확인
    if not Path(csv_path).exists():
        print(f"❌ 오류: {csv_path} 파일을 찾을 수 없습니다")
        print("   먼저 step1_clean_data.py를 실행하세요")
        return None

    # 데이터 로드
    print("📂 데이터 로드 중...", end=" ")
    df = pd.read_csv(csv_path)
    print(f"✅ ({len(df)}개 행, {df['battery'].nunique()}개 배터리)\n")

    eol_data = []

    print("🔄 배터리별 EOL 계산 중...\n")

    # 배터리별 처리
    for battery in sorted(df['battery'].unique()):
        batt_data = df[df['battery'] == battery].sort_values('cycle_num')

        # ===== Step 1: capacity_after_regen 찾기 =====
        # (최대 용량 = 배터리의 peak capacity)
        max_capacity_idx = batt_data['capacity_ahr'].idxmax()
        capacity_after_regen = batt_data.loc[max_capacity_idx, 'capacity_ahr']
        regen_cycles = int(batt_data.loc[max_capacity_idx, 'cycle_num'])

        # ===== Step 2: eol_threshold 계산 =====
        # (30% 감소 지점)
        eol_threshold = capacity_after_regen * 0.7

        # ===== Step 3: eol_dcycle 찾기 =====
        # (capacity_ahr ≤ threshold인 첫 번째 cycle)
        below_threshold = batt_data[batt_data['capacity_ahr'] <= eol_threshold]
        if len(below_threshold) > 0:
            eol_dcycle = int(below_threshold['cycle_num'].iloc[0])
        else:
            # threshold 이하로 내려가지 않으면 마지막 cycle
            eol_dcycle = int(batt_data['cycle_num'].max())

        # ===== Step 4: 추가 정보 =====
        capacity_deg = float(batt_data['capacity_deg'].iloc[0])
        cluster = int(batt_data['cluster'].iloc[0])
        final_capacity = float(batt_data['capacity_ahr'].iloc[-1])

        print(f"  {battery:>6s}: capacity={capacity_after_regen:.4f}, "
              f"threshold={eol_threshold:.4f}, eol_dcycle={eol_dcycle}")

        eol_data.append({
            'battery': battery,
            'capacity_after_regen': capacity_after_regen,
            'eol_threshold': eol_threshold,
            'eol_dcycle': eol_dcycle,
            'regen_cycles': regen_cycles,
            'final_capacity': final_capacity,
            'capacity_deg': capacity_deg,
            'cluster': cluster
        })

    eol_df = pd.DataFrame(eol_data)

    print("\n✅ EOL 계산 완료\n")

    return eol_df


def print_statistics(eol_df):
    """통계 정보 출력"""

    print("="*80)
    print("📊 통계")
    print("="*80 + "\n")

    print(f"총 배터리: {len(eol_df)}\n")

    print("Capacity after regen:")
    print(f"  최소: {eol_df['capacity_after_regen'].min():.4f} Ahr")
    print(f"  최대: {eol_df['capacity_after_regen'].max():.4f} Ahr")
    print(f"  평균: {eol_df['capacity_after_regen'].mean():.4f} Ahr\n")

    print("EOL_DCYCLE:")
    print(f"  최소: {eol_df['eol_dcycle'].min()} cycles")
    print(f"  최대: {eol_df['eol_dcycle'].max()} cycles")
    print(f"  평균: {eol_df['eol_dcycle'].mean():.1f} cycles")
    print(f"  중앙값: {eol_df['eol_dcycle'].median():.1f} cycles\n")

    print("Capacity degradation:")
    print(f"  최소: {eol_df['capacity_deg'].min():.2f}%")
    print(f"  최대: {eol_df['capacity_deg'].max():.2f}%")
    print(f"  평균: {eol_df['capacity_deg'].mean():.2f}%\n")

    print("Cluster:")
    print(f"  Cluster 0: {(eol_df['cluster'] == 0).sum()}개")
    print(f"  Cluster 1: {(eol_df['cluster'] == 1).sum()}개\n")

    # EOL 범위별 분류
    early_eol = eol_df[eol_df['eol_dcycle'] <= 6]
    mid_eol = eol_df[(eol_df['eol_dcycle'] > 6) & (eol_df['eol_dcycle'] <= 50)]
    late_eol = eol_df[eol_df['eol_dcycle'] > 50]

    print("EOL_DCYCLE 범위별 분류:")
    print(f"  1~6 cycles (조기 실패):      {len(early_eol):>2d}개")
    print(f"  7~50 cycles (중기 실패):     {len(mid_eol):>2d}개")
    print(f"  50+ cycles (정상 작동):      {len(late_eol):>2d}개\n")


def main():
    """메인 실행"""

    input_csv = '/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned.csv'
    output_csv = '/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_eol_info.csv'

    # EOL 정보 계산
    eol_df = calculate_eol_info(input_csv)

    if eol_df is None:
        return

    # 통계 출력
    print_statistics(eol_df)

    # 결과 미리보기
    print("="*80)
    print("📋 데이터 미리보기")
    print("="*80 + "\n")
    print(eol_df.to_string(index=False))

    # 저장
    eol_df.to_csv(output_csv, index=False)

    print("\n" + "="*80)
    print(f"✅ 저장 완료: {output_csv}")
    print("="*80)

    print("\n✅ 다음 단계:")
    print("   python3 step3_visualize.py")


if __name__ == "__main__":
    main()