import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 최종 62개 열 정의 (파이프라인 일관성을 위한 명시적인 목록)
# ============================================================================
TRAIN_COLUMNS_62 = [
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

# ============================================================================
# 파일 경로 설정 (고객님께서 제공하신 절대 경로 사용)
# ============================================================================
# ⚠️ 경로 수정: OUTPUT_FILE_TRAIN_LIGHT의 파일명을 (train)으로 수정했습니다.
# (이전 스크립트에서는 파일명이 (infer)로 되어 있었으나, 용도는 train light이므로 수정함)
FILE_TRAIN_62 = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/db/battery_training_data_cleaned_final.csv"
FILE_INFER_38 = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_38-2/output/(infer)battery_training_data_cleaned_final.csv"
OUTPUT_FILE_TRAIN_LIGHT = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/db/(train)battery_training_data_cleaned_final_38cols_light.csv"


# ============================================================================
# 1. 추론용 데이터셋을 로드하여 '유지할 컬럼 목록' (순서 포함) 추출
# ============================================================================
try:
    # 이 파일에서 38개 컬럼 목록 (순서대로)을 추출합니다.
    df_infer_38 = pd.read_csv(FILE_INFER_38)
    cols_to_keep = df_infer_38.columns.tolist() # 38개 열, 추론용의 정확한 순서
    
    # ⚠️ 안전성 검사: 추출된 38개 열이 62개 참조 목록에 모두 포함되는지 확인
    if not all(col in TRAIN_COLUMNS_62 for col in cols_to_keep):
        # 이전에 정의된 62개 목록에 없는 컬럼이 38개 목록에 있다면 심각한 오류입니다.
        print("❌ 치명적 오류: 38개 추론용 파일에 62개 학습용 파일과 호환되지 않는 컬럼이 포함되어 있습니다.")
        sys.exit(1)

    print("=" * 80)
    print(f"✅ 추론용 데이터셋 ({FILE_INFER_38})에서 38개 열 목록 추출 완료.")
    
except FileNotFoundError:
    print(f"❌ 오류: 참조용 추론 파일 {FILE_INFER_38}을 찾을 수 없습니다. 경로를 확인하십시오.")
    sys.exit(1)

# ============================================================================
# 2. 학습용 데이터셋을 로드하고 38개 컬럼만 남기고 서브셋 생성
# ============================================================================
try:
    df_train_62 = pd.read_csv(FILE_TRAIN_62)
    
    # 38개 컬럼만 선택하고, 순서를 'cols_to_keep' (추론용의 순서)와 동일하게 맞춥니다.
    # df_train_62[cols_to_keep] 명령이 이 작업을 수행합니다.
    df_train_light = df_train_62[cols_to_keep].copy()

    # ========================================================================
    # 3. 새로운 라이트 버전 학습 데이터셋 저장
    # ========================================================================
    df_train_light.to_csv(OUTPUT_FILE_TRAIN_LIGHT, index=False)
    
    print(f"✅ [SUCCESS] 'Final' 학습 데이터의 라이트 버전 생성 완료.")
    print(f"  - 원본 학습 (62개 열): {FILE_TRAIN_62}")
    print(f"  - 새 학습 (38개 열):  {OUTPUT_FILE_TRAIN_LIGHT}")
    print(f"  - 새 파일 경로: {Path(OUTPUT_FILE_TRAIN_LIGHT).parent}")
    print("=" * 80)
    
except FileNotFoundError:
    print(f"❌ 오류: 학습 파일 {FILE_TRAIN_62}을 찾을 수 없습니다. 경로를 확인하십시오.")
except Exception as e:
    print(f"❌ 치명적인 오류 발생: {e}")