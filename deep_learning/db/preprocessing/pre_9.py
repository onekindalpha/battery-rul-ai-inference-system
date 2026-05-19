"""
Battery Training Dataset 전처리 코드
모든 문제 해결: 극단값, 음수값, 결측치, 0값
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_battery_data(csv_path, output_path=None):
    """
    완전한 전처리 파이프라인

    Args:
        csv_path (str): 입력 CSV 파일 경로
        output_path (str): 출력 CSV 파일 경로 (None이면 저장 안 함)

    Returns:
        pd.DataFrame: 전처리된 데이터
    """

    print("=" * 80)
    print("🔄 배터리 데이터 전처리 시작")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n【 Step 1: 데이터 로드 】")
    df = pd.read_csv(csv_path)
    print(f"✓ 로드 완료: {df.shape[0]} 행, {df.shape[1]} 컬럼")

    # 2. capacity_ahr = 0 제거
    print("\n【 Step 2: capacity_ahr = 0 제거 】")
    zero_count = (df['capacity_ahr'] == 0).sum()
    df = df[df['capacity_ahr'] > 0].copy()
    print(f"✓ {zero_count}개 행 제거")
    print(f"  행 수: {len(df)}")

    # 3. 결측치 처리
    print("\n【 Step 3: 결측치 처리 】")
    print(f"처리 전 결측치 합계: {df.isnull().sum().sum()}개")

    # 배터리별 중앙값으로 fillna
    for col in ['capacity_derivative', 'cycle_diff_time', 'cycle_duration_rate']:
        before = df[col].isnull().sum()
        df[col] = df.groupby('battery')[col].transform(
            lambda x: x.fillna(x.median())
        )
        after = df[col].isnull().sum()
        print(f"✓ {col}: {before} → {after}")

    # 여전히 남은 결측치 제거 (특수한 경우)
    remaining_null = df.isnull().sum().sum()
    if remaining_null > 0:
        print(f"  남은 결측치: {remaining_null}개 제거")
        df = df.dropna()

    print(f"처리 후 결측치: {df.isnull().sum().sum()}개")

    # 4. 극단값 처리
    print("\n【 Step 4: 극단값 처리 】")

    # 4-1. dcr_growth clipping
    print("4-1) dcr_growth 극단값 해결")
    print(f"  처리 전: {df['dcr_growth'].min():.2e} ~ {df['dcr_growth'].max():.2e}")
    df['dcr_growth'] = np.clip(df['dcr_growth'], -1000, 1000)
    print(f"  처리 후: {df['dcr_growth'].min():.2f} ~ {df['dcr_growth'].max():.2f}")
    print(f"  ✓ clipping으로 정규화")

    # 4-2. 기타 이상치 처리 (선택적)
    for col in ['impedance_growth', 'impedance_sum']:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        before = ((df[col] < q1) | (df[col] > q99)).sum()
        if before > 0:
            df[col] = np.clip(df[col], q1, q99)
            print(f"✓ {col}: {before}개 이상치 clipping")

    # 5. 음수값 처리 (절댓값 생성)
    print("\n【 Step 5: 음수값 처리 】")
    print("음수값은 정상 (방전 정의)")

    # 절댓값 버전 생성
    df['C_rate_abs'] = np.abs(df['C_rate_avg'])
    df['C_rate_max_abs'] = np.abs(df['C_rate_max'])
    df['current_temp_product_abs'] = np.abs(df['current_temp_product'])
    df['load_temp_interact_abs'] = np.abs(df['load_temp_interact'])
    df['ir_drop_abs'] = np.abs(df['ir_drop'])

    print(f"✓ 절댓값 컬럼 5개 생성:")
    print(f"  - C_rate_abs: {df['C_rate_abs'].min():.4f} ~ {df['C_rate_abs'].max():.4f}")
    print(f"  - current_temp_product_abs: {df['current_temp_product_abs'].min():.2f} ~ {df['current_temp_product_abs'].max():.2f}")
    print(f"  - load_temp_interact_abs: {df['load_temp_interact_abs'].min():.2f} ~ {df['load_temp_interact_abs'].max():.2f}")
    print(f"  - ir_drop_abs: {df['ir_drop_abs'].min():.4f} ~ {df['ir_drop_abs'].max():.4f}")

    # 6. 로그 변환 (선택적, 매우 큰 값)
    print("\n【 Step 6: 로그 변환 (선택적) 】")
    df['impedance_growth_log'] = np.sign(df['impedance_growth']) * np.log(1 + np.abs(df['impedance_growth']))
    df['dcr_growth_log'] = np.sign(df['dcr_growth']) * np.log(1 + np.abs(df['dcr_growth']))
    print(f"✓ 로그 변환 컬럼 2개 생성 (큰 범위의 값들을 압축)")

    # 7. 불필요한 컬럼 제거 (선택적)
    print("\n【 Step 7: 컬럼 최적화 】")
    # soh와 capacity_norm은 동일 → 하나만 유지
    if 'capacity_norm' in df.columns:
        df = df.drop('capacity_norm', axis=1)
        print(f"✓ capacity_norm 제거 (soh와 중복)")

    # 8. 최종 검증
    print("\n【 Step 8: 최종 검증 】")
    print(f"최종 행 수: {len(df)}")
    print(f"최종 컬럼 수: {len(df.columns)}")
    print(f"결측치: {df.isnull().sum().sum()}개 ✓")
    print(f"데이터 타입: {df.dtypes.value_counts().to_dict()}")

    # 통계 요약
    print("\n【 주요 컬럼 통계 】")
    key_cols = ['capacity_ahr', 'capacity_after_regen', 'soh', 'dcr', 'lli', 'lam',
                'C_rate_abs', 'temp_rise', 'impedance_growth']
    for col in key_cols:
        if col in df.columns:
            print(f"{col}:")
            print(f"  범위: {df[col].min():.4f} ~ {df[col].max():.4f}")
            print(f"  평균: {df[col].mean():.4f}")

    # 9. 저장 (선택적)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n【 저장 완료 】")
        print(f"✓ {output_path}")

    print("\n" + "=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)

    return df

def get_train_test_split(df, test_batteries=None, random_state=42):
    """
    배터리별로 train/test 분리

    Args:
        df (pd.DataFrame): 전처리된 데이터
        test_batteries (list): 테스트 배터리 ID (None이면 마지막 2개)
        random_state (int): 난수 시드

    Returns:
        tuple: (df_train, df_test)
    """

    all_batteries = sorted(df['battery'].unique())

    if test_batteries is None:
        test_batteries = all_batteries[-2:]

    train_batteries = [b for b in all_batteries if b not in test_batteries]

    df_train = df[df['battery'].isin(train_batteries)].copy()
    df_test = df[df['battery'].isin(test_batteries)].copy()

    print(f"Train: {len(train_batteries)} 배터리, {len(df_train)} 행")
    print(f"Test: {len(test_batteries)} 배터리, {len(df_test)} 행")

    return df_train, df_test

def normalize_features(df_train, df_test, exclude_cols=None):
    """
    StandardScaler로 정규화

    Args:
        df_train (pd.DataFrame): 훈련 데이터
        df_test (pd.DataFrame): 테스트 데이터
        exclude_cols (list): 정규화 제외 컬럼

    Returns:
        tuple: (df_train_norm, df_test_norm)
    """

    if exclude_cols is None:
        exclude_cols = ['battery', 'battery_encoded', '_crate_bin', '_temp_bin', 'cluster']

    # 정규화할 컬럼 선택
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]

    # StandardScaler 학습
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    df_train_scaled[cols_to_normalize] = scaler.fit_transform(df_train[cols_to_normalize])
    df_test_scaled[cols_to_normalize] = scaler.transform(df_test[cols_to_normalize])

    print(f"✓ {len(cols_to_normalize)}개 컬럼 정규화 완료")

    return df_train_scaled, df_test_scaled


# ============================================================================
# 사용 예제
# ============================================================================

if __name__ == "__main__":

    # 전처리 실행
    df_clean = preprocess_battery_data(
        csv_path='/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project//db/nasabattery/output/battery_training_data_completed.csv',
        output_path='/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned_final.csv'
    )

    print("\n\n")