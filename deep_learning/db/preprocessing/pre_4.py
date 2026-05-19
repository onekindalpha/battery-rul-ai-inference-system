"""
1단계: 배터리 데이터 정제
========================================
원본 CSV에서 B0050, B0052를 제외한 깨끗한 데이터 생성

입력: battery_final_training_data.csv (원본)
출력: battery_training_data_cleaned.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_battery_data(input_csv, output_csv):
    """
    배터리 데이터 정제
    - B0050, B0052 제외 (결측치 많음)
    - 정제된 CSV 저장

    Parameters:
    -----------
    input_csv : str
        원본 배터리 데이터 파일
    output_csv : str
        정제된 배터리 데이터 파일 (저장 경로)
    """

    print("="*80)
    print("1단계: 배터리 데이터 정제")
    print("="*80 + "\n")

    # 파일 확인
    if not Path(input_csv).exists():
        print(f"❌ 오류: {input_csv} 파일을 찾을 수 없습니다")
        return False

    # 데이터 로드
    print("📂 데이터 로드 중...", end=" ")
    df = pd.read_csv(input_csv)
    print(f"✅ ({len(df)}개 행, {df['battery'].nunique()}개 배터리)\n")

    # 결측치 현황
    print("📊 원본 데이터 결측치:")
    null_summary = df.isnull().sum()
    if null_summary.sum() > 0:
        print(f"  총 {null_summary.sum()}개 결측치")
        for col, count in null_summary[null_summary > 0].items():
            print(f"    - {col}: {count}개")
    else:
        print("  없음")

    # 제외할 배터리
    exclude_batteries = ['B0050', 'B0052']

    print(f"\n🗑️  제외할 배터리: {', '.join(exclude_batteries)}")
    excluded_rows = df[df['battery'].isin(exclude_batteries)]
    print(f"  제외 행 수: {len(excluded_rows)}개\n")

    # 정제
    df_clean = df[~df['battery'].isin(exclude_batteries)].copy()

    print("✅ 정제 완료:")
    print(f"  원본: {len(df)}개 행, {df['battery'].nunique()}개 배터리")
    print(f"  정제: {len(df_clean)}개 행, {df_clean['battery'].nunique()}개 배터리\n")

    # 정제된 데이터 결측치 확인
    clean_nulls = df_clean.isnull().sum()
    if clean_nulls.sum() > 0:
        print("⚠️  정제 후 결측치:")
        for col, count in clean_nulls[clean_nulls > 0].items():
            print(f"    - {col}: {count}개")
    else:
        print("✅ 정제 후 결측치: 없음\n")

    # 저장
    df_clean.to_csv(output_csv, index=False)

    print("="*80)
    print(f"✅ 저장 완료: {output_csv}")
    print("="*80)

    return True


def main():
    """메인 실행"""

    input_csv = '/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_final_training_data.csv'
    output_csv = '/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned.csv'

    success = clean_battery_data(input_csv, output_csv)

    if success:
        print("\n✅ 다음 단계:")
        print("   python3 generate_eol_info.py")


if __name__ == "__main__":
    main()