import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ⚠️ 주의: 이 스크립트는 원본 sensor 데이터(min/max/std 등)가 38개 열 파일에 없으므로,
# 해당 데이터를 기반으로 만들어지는 24개 파생 피처를 '근사 처리'하거나 '0'으로 채웁니다.
# 이는 모델의 추론 정확도를 떨어뜨릴 수 있습니다. 이상적인 해결책은 원본 sensor 데이터부터
# 전체 학습 파이프라인(pre_1.py ~ pre_8.py)을 동일하게 재실행하는 것입니다.

# ============================================================================
# 학습용 데이터의 62개 열 순서 (추론 데이터의 열 순서 보장을 위한 기준)
# ============================================================================
TRAIN_COLUMNS = [
    'battery', 'cycle_index', 'cycle_num', 'time_seconds', 'capacity_ahr',
    'ambient_temp_c', 'voltage_measured_mean', 'voltage_measured_min', 'voltage_measured_max',
    'current_measured_mean', 'current_measured_std', 'temperature_measured_mean',
    'temperature_measured_max', 'current_load_mean', 'voltage_load_mean',
    'discharge_time_sec', 're_ohm_interp', 'rct_ohm_interp', 'sense_current_interp',
    'battery_current_interp', 'current_ratio_interp', 'battery_impedance_interp',
    'rectified_impedance_interp', 'regen_cycles', 'capacity_after_regen', 'capacity_deg',
    'cluster', 'dcr', 'lli', 'lam', 'soh', 'capacity_derivative', 'regen_strength',
    'impedance_sum', 'impedance_growth', 'dcr_growth', 'lli_scaled', 'lli_smooth',
    'lam_scaled', 'lam_smooth', 'dcr_scaled', 'dcr_smooth', 'temp_rise', 'thermal_stress',
    'ir_drop', 'voltage_range', 'C_rate_avg', 'C_rate_max', 'current_temp_product',
    'load_temp_interact', 'cycle_diff_time', 'cycle_duration_rate', 'battery_encoded',
    '_crate_bin', '_temp_bin', 'C_rate_abs', 'C_rate_max_abs', 'current_temp_product_abs',
    'load_temp_interact_abs', 'ir_drop_abs', 'impedance_growth_log', 'dcr_growth_log'
]

def engineer_inference_features(df_infer):
    df_copy = df_infer.copy()

    # ========================================================================
    # 1. 누락된 원시/중간 열에 대한 임시 값 할당 (24개 열 생성의 전제 조건)
    # ========================================================================

    # 'cycle_index' 생성 (cycle_num과 동일하다고 가정)
    df_copy['cycle_index'] = df_copy['cycle_num']

    # 누락된 핵심 원시/중간 열에 대해 임시값 (0.0)을 할당합니다.
    # 이 열들은 38개 파일에 없으므로, 원칙적으로는 원시 데이터에서 추출해야 합니다.
    intermediate_to_add = [
        'voltage_measured_min', 'voltage_measured_max', 'current_measured_mean',
        'current_measured_std', 'temperature_measured_mean', 'current_load_mean',
        'voltage_load_mean', 'sense_current_interp', 'battery_current_interp',
        'current_ratio_interp', 'rectified_impedance_interp'
    ]
    for col in intermediate_to_add:
        if col not in df_copy.columns:
            df_copy[col] = 0.0 # ⚠️ 모델에 입력될 때 심각한 오류를 피하기 위한 임시 조치

    # ========================================================================
    # 2. 파생 피처 재계산 (pre_7.py 및 pre_8.py 로직 통합)
    # ========================================================================

    # 2-1. 임피던스 성장 관련 (38개 파일에 있는 dcr, impedance_growth를 다시 계산)
    df_copy['impedance_sum'] = df_copy['re_ohm_interp'] + df_copy['rct_ohm_interp']
    initial_impedance = df_copy.groupby('battery')['impedance_sum'].transform('first')
    df_copy['impedance_growth'] = ((df_copy['impedance_sum'] - initial_impedance) / (initial_impedance + 1e-9) * 100).fillna(0)
    
    initial_dcr = df_copy.groupby('battery')['dcr'].transform('first')
    df_copy['dcr_growth'] = ((df_copy['dcr'] - initial_dcr) / (initial_dcr + 1e-9) * 100).fillna(0)

    # 2-2. 용량 관련 (soh, capacity_derivative, regen_strength)
    df_copy['soh'] = df_copy['capacity_ahr'] / (df_copy['capacity_after_regen'] + 1e-9)
    df_copy['capacity_derivative'] = df_copy.groupby('battery')['capacity_ahr'].diff().fillna(0)
    df_copy['regen_strength'] = df_copy['capacity_after_regen'] - df_copy['capacity_ahr']
    
    # 2-3. 열화 지표 스케일링 & 평활 (lli, lam 등 38개 파일에 있는 열화 지표 사용)
    for col in ['lli', 'lam', 'dcr']:
        # 38개 파일에 없는 경우 0으로 채움
        if col not in df_copy.columns: df_copy[col] = 0.0
        
        # 학습 데이터의 Min/Max/Std를 알 수 없으므로, 현재 추론 데이터의 통계값으로 스케일링
        # ⚠️ (모델을 학습시킨 스케일러의 통계값 대신 현재 데이터 통계값 사용 - 성능 저하 요인)
        mean = df_copy[col].mean()
        std = df_copy[col].std()
        df_copy[f'{col}_scaled'] = np.clip((df_copy[col] - mean) / (std + 1e-9), -3, 3)
        df_copy[f'{col}_smooth'] = df_copy.groupby('battery')[col].transform(lambda x: x.rolling(window=5, center=True, min_periods=1).mean())

    # 2-4. 스트레스/전기적 파생 (누락된 24개 열의 핵심)
    df_copy['temp_rise'] = df_copy['temperature_measured_max'] - df_copy['ambient_temp_c']
    df_copy['thermal_stress'] = df_copy['temp_rise'] / (df_copy['discharge_time_sec'] + 1e-9)
    df_copy['ir_drop'] = df_copy['voltage_load_mean'] - df_copy['voltage_measured_mean']
    df_copy['voltage_range'] = df_copy['voltage_measured_max'] - df_copy['voltage_measured_min']
    
    df_copy['C_rate_avg'] = df_copy['current_measured_mean'] / (df_copy['capacity_after_regen'] + 1e-9)
    df_copy['C_rate_max'] = (df_copy['current_measured_mean'] + 2 * df_copy['current_measured_std']) / (df_copy['capacity_after_regen'] + 1e-9)
    df_copy['current_temp_product'] = df_copy['current_measured_mean'] * df_copy['temp_rise']
    df_copy['load_temp_interact'] = df_copy['C_rate_avg'] * df_copy['temp_rise']

    # 2-5. 시간/사이클 파생
    df_copy['cycle_diff_time'] = df_copy.groupby('battery')['time_seconds'].diff().fillna(0)
    df_copy['cycle_duration_rate'] = df_copy['cycle_diff_time'] / (df_copy['discharge_time_sec'] + 1e-9)
    
    # 2-6. 인코딩 & 구간화
    le = LabelEncoder()
    df_copy['battery_encoded'] = le.fit_transform(df_copy['battery'])
    # 학습 데이터의 bin 기준을 알 수 없으므로, 현재 데이터 기준으로 재구간화
    df_copy['_crate_bin'] = pd.cut(df_copy['C_rate_avg'], bins=5, labels=False, duplicates='drop').fillna(-1).astype(int)
    # df_copy['_temp_bin']은 38개 파일에 이미 존재한다고 가정

    # 2-7. 절대값 및 로그 변환
    for col in ['C_rate_avg', 'C_rate_max', 'current_temp_product', 'load_temp_interact', 'ir_drop']:
        df_copy[f'{col}_abs'] = np.abs(df_copy[col])
        
    df_copy['impedance_growth_log'] = np.sign(df_copy['impedance_growth']) * np.log(1 + np.abs(df_copy['impedance_growth']))
    df_copy['dcr_growth_log'] = np.sign(df_copy['dcr_growth']) * np.log(1 + np.abs(df_copy['dcr_growth']))

    # 3. 최종 컬럼 순서 맞추기
    # 모든 열이 생성되었는지 확인 후, TRAIN_COLUMNS의 순서에 맞춰 열을 재배열합니다.
    missing_cols_after_engineering = [col for col in TRAIN_COLUMNS if col not in df_copy.columns]
    for col in missing_cols_after_engineering:
        df_copy[col] = 0.0
        
    df_processed = df_copy[TRAIN_COLUMNS]

    return df_processed

# ============================================================================
# 최종 실행
# ============================================================================

INPUT_FILE = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_38-2/output/(infer)battery_training_data_cleaned_final.csv"
OUTPUT_FILE = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_38-2/output/(infer)battery_training_data_cleaned_final_62cols_fixed.csv"

try:
    df_infer_38 = pd.read_csv(INPUT_FILE)
    df_infer_62 = engineer_inference_features(df_infer_38)
    
    df_infer_62.to_csv(OUTPUT_FILE, index=False)
    
    print("=" * 80)
    print(f"✅ [SUCCESS] {INPUT_FILE} ({len(df_infer_38.columns)} cols) 파일을")
    print(f"            학습용 데이터셋과 호환되는 {len(df_infer_62.columns)}개 열로 변환 완료.")
    print(f"            새 파일: {OUTPUT_FILE}")
    print("=" * 80)

except FileNotFoundError:
    print(f"❌ 오류: 입력 파일 {INPUT_FILE}을 찾을 수 없습니다. 파일 이름을 확인하십시오.")
except Exception as e:
    print(f"❌ 오류 발생: {e}")