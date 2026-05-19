import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# 파일 경로 설정 (기존 파일명 사용)
# ============================================================================
FILE_TRAIN_62 = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/db/(train)battery_training_data_cleaned_final.csv" # 62개 열 학습용 (원본)
FILE_INFER_38 = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_38-2/output/(infer)battery_training_data_cleaned_final.csv" # 38개 열 추론용 (참조용)
OUTPUT_FILE_TRAIN_LIGHT = "/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_38-2/output/(infer)battery_training_data_cleaned_final_38cols_light.csv"

# ============================================================================
# 1. 추론용 데이터셋을 로드하여 사용할 38개 컬럼 목록 추출
# ============================================================================
try:
    df_infer_38 = pd.read_csv(FILE_INFER_38)
    cols_to_keep = df_infer_38.columns.tolist()
    
    print("=" * 80)
    print(f"✅ 추론용 데이터셋 ({FILE_INFER_38})에서 38개 열 목록 추출 완료.")
    print(f"   - 추출된 열 개수: {len(cols_to_keep)}")
    
except FileNotFoundError:
    print(f"❌ 오류: 참조용 추론 파일 {FILE_INFER_38}을 찾을 수 없습니다. 파일 경로를 확인하십시오.")
    sys.exit(1)

# ============================================================================
# 2. 학습용 데이터셋을 로드하고 38개 컬럼만 남기고 서브셋 생성
# ============================================================================
try:
    df_train_62 = pd.read_csv(FILE_TRAIN_62)
    
    # 38개 컬럼이 62개 파일에 모두 있는지 확인 (안전성 보장)
    if not all(col in df_train_62.columns for col in cols_to_keep):
        missing_cols = [col for col in cols_to_keep if col not in df_train_62.columns]
        print(f"⚠️ 경고: 학습용 파일에 추론용 파일의 컬럼 중 {len(missing_cols)}개가 없습니다. -> {missing_cols}")
        
        # 누락된 컬럼만 제거하고 진행
        cols_to_keep = [col for col in cols_to_keep if col in df_train_62.columns]
        if not cols_to_keep:
            raise ValueError("남은 공통 컬럼이 없어 라이트 버전 생성에 실패했습니다.")

    # 38개 컬럼만 선택하고, 순서를 추론용과 동일하게 맞춥니다.
    df_train_light = df_train_62[cols_to_keep].copy()

    # ========================================================================
    # 3. 새로운 라이트 버전 학습 데이터셋 저장
    # ========================================================================
    df_train_light.to_csv(OUTPUT_FILE_TRAIN_LIGHT, index=False)
    
    print(f"✅ [SUCCESS] 'Final' 학습 데이터의 라이트 버전 생성 완료.")
    print(f"  - 원본 학습 (62개 열): {FILE_TRAIN_62}")
    print(f"  - 새 학습 (38개 열):  {OUTPUT_FILE_TRAIN_LIGHT}")
    print("=" * 80)
    
except FileNotFoundError:
    print(f"❌ 오류: 학습 파일 {FILE_TRAIN_62}을 찾을 수 없습니다. 파일 경로를 확인하십시오.")
except Exception as e:
    print(f"❌ 치명적인 오류 발생: {e}")