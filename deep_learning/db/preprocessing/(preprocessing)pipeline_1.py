"""
배터리 데이터셋 엔드-투-엔드 통합 파이프라인 스크립트

1. 원본 XLSX 데이터 로드 및 통합 (pre_1.py 기능)
2. 임피던스 외삽 (pre_2.py 기능)
3. 열화 지표 계산 & 클러스터링 (pre_3.py 기능)
4. 데이터 정제 (pre_4.py 기능)
5. DCR, LLi, LAM 추가 (pre_7.py 기능)
6. EOL 정보 계산 (pre_5.py 기능)
7. 광범위한 피처 엔지니어링 (pre_10.py 기능)
8. 최종 정제 및 학습 데이터 완성 (pre_9.py 기능) -> battery_training_data_cleaned_final.csv 생성
9. Advanced CEEMDAN 분해 (ceemdan_3.py 기능) -> ceemdan_all_batteries_advanced.csv 생성
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import argrelextrema, find_peaks
from pathlib import Path
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# ============================================================================
# 0. 설정 및 경로
# ============================================================================

# 사용자 지정 경로로 BASE_DIR 및 INPUT_DIR 직접 설정
# NASABATTERY 폴더가 INPUT_DIR의 부모 디렉토리에 있는 구조로 가정합니다.
# 수정된 경로: /Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_35-2
# 원본 데이터(.xlsx)가 'CS2_35-2' 폴더 안에 직접 있다면, 아래와 같이 설정합니다.

# 원본 데이터 폴더를 INPUT_DIR로 직접 지정합니다.
INPUT_DIR_ROOT = Path('/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/CS2_38-2')
INPUT_DIR = INPUT_DIR_ROOT # 원본 .xlsx 파일이 있는 디렉토리
OUTPUT_DIR = INPUT_DIR_ROOT / 'output' # 결과물을 저장할 디렉토리는 INPUT_DIR_ROOT 내부에 생성

# 기존 BASE_DIR을 제거하거나, OUTPUT_DIR을 INPUT_DIR_ROOT와 동일한 레벨에 생성할 경우
# BASE_DIR = Path('/Users/velocitygoal/Desktop/battery_project/v10.4/deep_learning/inference/cacle_dataset/')
# INPUT_DIR = BASE_DIR / 'CS2_35-2'
# OUTPUT_DIR = BASE_DIR / 'output'
# 와 같이 설정할 수도 있습니다. (선택)
# 여기서는 INPUT_DIR_ROOT 내부에 output을 생성하는 첫 번째 방법을 사용합니다.

OUTPUT_DIR.mkdir(exist_ok=True) 

# 통합 스크립트에서 사용할 파일 이름 (중간 저장용)
FILE_COMPLETE_DATA = OUTPUT_DIR / 'battery_complete_data.csv'
FILE_COMPLETE_FINAL = OUTPUT_DIR / 'battery_complete_data_final.csv'
FILE_FINAL_TRAINING = OUTPUT_DIR / 'battery_final_training_data.csv'
FILE_CLEANED_V1 = OUTPUT_DIR / 'battery_training_data_cleaned.csv'
FILE_EOL_INFO = OUTPUT_DIR / 'battery_eol_info.csv'
FILE_COMPLETED = OUTPUT_DIR / 'battery_training_data_completed.csv'
FILE_CLEANED_FINAL = OUTPUT_DIR / 'battery_training_data_cleaned_final.csv' # 목표 1
FILE_CEEMDAN_FINAL = OUTPUT_DIR / 'ceemdan_all_batteries_advanced.csv'    # 목표 2

print("=" * 80)
print("🔋 End-to-End 배터리 데이터 파이프라인 시작")
print(f"✅ INPUT_DIR 설정: {INPUT_DIR}")
print(f"✅ OUTPUT_DIR 설정: {OUTPUT_DIR}")
print("=" * 80)
# ============================================================================
# Advanced CEEMDAN 클래스 (Step 9용)
# ============================================================================

class AdvancedCEEMDAN:
    """실제 신호 분해를 수행하는 고급 CEEMDAN 클래스"""
    def __init__(self, noise_mode='adaptive', n_ensembles=150, seed=42):
        self.noise_mode = noise_mode
        self.n_ensembles = n_ensembles
        np.random.seed(seed)
        
    # (ceemdan_3.py의 analyze_signal_characteristics, set_adaptive_noise, 
    # find_extrema_advanced, get_smooth_envelope, sift_advanced, decompose 메서드는 
    # 코드가 길어지므로 생략하나, 실제 통합시에는 클래스 안에 모두 복사되어야 함)
    # (편의상 여기서는 핵심 로직만 남깁니다.)

    def analyze_signal_characteristics(self, signal):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_normalized = (signal - signal_mean) / (signal_std + 1e-10)
        signal_diff = np.diff(signal_normalized)
        volatility = np.std(signal_diff)
        skewness = np.abs(np.mean(signal_diff ** 3) / (volatility ** 3 + 1e-10))
        return {'mean': signal_mean, 'std': signal_std, 'normalized': signal_normalized, 'volatility': volatility, 'skewness': skewness}

    def set_adaptive_noise(self, signal_chars):
        volatility = signal_chars['volatility']
        skewness = signal_chars['skewness']
        base_noise = 0.2 * volatility
        base_noise *= (1 + 0.3 * np.tanh(skewness))
        return np.clip(base_noise, 0.01, 0.3)
    
    def find_extrema_advanced(self, signal, order=2):
        if len(signal) < 5: return np.array([], dtype=int), np.array([], dtype=int)
        maxima_idx = argrelextrema(signal, np.greater, order=order)[0]
        minima_idx = argrelextrema(signal, np.less, order=order)[0]
        # 경계값 및 극값 적을 때 처리 로직 (pre_3.py와 동일하게 처리)
        # ...
        return maxima_idx, minima_idx

    def get_smooth_envelope(self, signal, extrema_idx, kind='max'):
        if len(extrema_idx) < 2: return np.full_like(signal, np.max(signal) if kind == 'max' else np.min(signal))
        try: return CubicSpline(extrema_idx, signal[extrema_idx], bc_type='natural', extrapolate='linear')(np.arange(len(signal)))
        except: return np.interp(np.arange(len(signal)), extrema_idx, signal[extrema_idx])

    def sift_advanced(self, signal, max_iterations=200):
        residual = signal.copy()
        for iteration in range(max_iterations):
            maxima_idx, minima_idx = self.find_extrema_advanced(residual, order=2)
            if len(maxima_idx) < 2 or len(minima_idx) < 2: break
            upper_env = self.get_smooth_envelope(residual, maxima_idx, 'max')
            lower_env = self.get_smooth_envelope(residual, minima_idx, 'min')
            mean_env = (upper_env + lower_env) / 2
            residual_new = residual - mean_env
            sd = np.sum((residual_new - residual) ** 2) / (np.sum(residual ** 2) + 1e-10)
            if sd < 0.0001: break
            residual = residual_new
        return residual
    
    def decompose(self, signal, max_imfs=6):
        signal_chars = self.analyze_signal_characteristics(signal)
        signal_normalized = signal_chars['normalized']
        signal_std = signal_chars['std']
        
        noise_std = self.set_adaptive_noise(signal_chars) if self.noise_mode == 'adaptive' else 0.2
        n_samples = len(signal_normalized)
        imfs_raw = []
        residual = signal_normalized.copy()
        
        for mode_idx in range(max_imfs):
            if np.std(residual) < 0.01 * np.std(signal_normalized): break
            imf_ensemble = np.zeros((self.n_ensembles, n_samples))
            for ens_idx in range(self.n_ensembles):
                noise = np.random.normal(0, noise_std, n_samples)
                signal_with_noise = residual + noise
                imf_candidate = self.sift_advanced(signal_with_noise, max_iterations=200)
                imf_ensemble[ens_idx] = imf_candidate
            
            imf = np.mean(imf_ensemble, axis=0)
            imfs_raw.append(imf)
            residual = residual - imf
        
        imfs = [imf * signal_std for imf in imfs_raw]
        residual = residual * signal_std + signal_chars['mean']
        while len(imfs) < 6: imfs.append(np.zeros_like(signal))
        return imfs, residual

# ============================================================================
# 1. Step 1: 원시 데이터 추출 및 1차 보간 (pre_1.py 기능)
# ============================================================================

def step1_extract_and_interpolate():
    print("\n" + "=" * 25 + " Step 1: XLSX 로드 및 1차 보간 " + "=" * 25)
    
    excel_files_lower = list(INPUT_DIR.glob('*.xlsx'))
    excel_files_upper = list(INPUT_DIR.glob('*.XLSX'))
    excel_files = sorted(excel_files_lower + excel_files_upper)
    
    if not excel_files:
        print(f"❌ .xlsx 파일을 찾을 수 없습니다! 디렉토리 확인: {INPUT_DIR}")
        return False
    
    print(f"✅ 총 {len(excel_files)}개의 .xlsx 파일 발견. 첫 3개: {[f.name for f in excel_files[:3]]}")

    all_integrated_data = []

    for excel_file in excel_files:
        battery_id = excel_file.stem 
        
        try:
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            
            # 시트 탐색: Statistics 또는 Channel
            target_sheet_name = next((s for s in excel_data if 'Statistics' in s or 'Channel' in s), None)

            if target_sheet_name:
                df_d = pd.DataFrame()
                header_found = False
                
                # 헤더 탐색 (0~9행 시도)
                for h_idx in range(10):
                    try:
                        temp_df = pd.read_excel(excel_file, sheet_name=target_sheet_name, header=h_idx)
                        # 유효 헤더 판단 기준 확장
                        if len(temp_df) > 1:
                            cols = [str(c).lower() for c in temp_df.columns]
                            if any('cycle' in c for c in cols) or any('time' in c for c in cols):
                                df_d = temp_df.copy()
                                header_found = True
                                break
                    except:
                        continue
                
                if not header_found:
                     # 마지막 시도로 헤더 없이 로드 후 첫 행을 헤더로 가정해보기
                    try:
                         df_d = pd.read_excel(excel_file, sheet_name=target_sheet_name, header=None)
                         if len(df_d) > 5:
                             # 문자열이 많은 행을 헤더로 간주하는 로직 등을 추가할 수 있으나 복잡해짐.
                             # 여기서는 일단 실패로 처리.
                             pass
                    except:
                        pass

                if df_d.empty:
                     raise ValueError(f"유효한 데이터 시트('{target_sheet_name}')를 찾았으나 헤더 인식에 실패했습니다.")

                df_d['battery'] = battery_id
                
                # ⚠️ 컬럼 매핑 강화 (대소문자 공백 무시 매핑 시도)
                # 원본 컬럼명을 표준화하여 매핑 성공률을 높입니다.
                df_d.columns = df_d.columns.astype(str).str.strip()
                
                rename_map = {
                    'Cycle_Index': 'cycle_num', 'Cycle Index': 'cycle_num',
                    'Discharge_Capacity(Ah)': 'capacity_ahr', 'Discharge Capacity(Ah)': 'capacity_ahr',
                    'Voltage(V)': 'voltage_measured_mean', 'Voltage': 'voltage_measured_mean',
                    'Internal_Resistance(Ohm)': 're_ohm_interp', 'Internal Resistance(Ohm)': 're_ohm_interp',
                    'AC_Impedance(Ohm)': 'battery_impedance_mean_ohm_interp',
                    'Date_Time': 'time_seconds', 'DateTime': 'time_seconds',
                    'DisCharge_Time(s)': 'discharge_time_sec', 'Discharge Time(s)': 'discharge_time_sec', 'Test_Time(s)': 'discharge_time_sec', # Test_Time을 대체제로 사용
                    'Vmax_On_Cycle(V)': 'voltage_measured_max'
                }
                df_d = df_d.rename(columns=rename_map)

                # Rct_ohm_interp 처리
                if 're_ohm_interp' in df_d.columns:
                    df_d['rct_ohm_interp'] = df_d['re_ohm_interp']
                
                # ⚠️ 핵심 수정: 파이프라인에 필요한 모든 컬럼을 강제로 보장합니다.
                # 원본 데이터에 없으면 NaN으로 채워 나중 단계에서의 KeyError를 방지합니다.
                required_cols_all = [
                    'battery', 'cycle_num', 'capacity_ahr', 'ambient_temp_c', 
                    'voltage_measured_mean', 'temperature_measured_mean', 'discharge_time_sec', 
                    'time_seconds', 'current_measured_mean', 'voltage_load_mean', 
                    'temperature_measured_max', 'voltage_measured_max', 'voltage_measured_min', 
                    'current_measured_std', 're_ohm_interp', 'rct_ohm_interp', 
                    'battery_impedance_mean_ohm_interp'
                ]
                
                for col in required_cols_all:
                    if col not in df_d.columns:
                        df_d[col] = np.nan

                # cycle_num이 NaN인 행 제거 (유효하지 않은 데이터)
                if 'cycle_num' in df_d.columns:
                     df_d = df_d.dropna(subset=['cycle_num'])

                # 필요한 컬럼만 선택하여 데이터프레임 생성
                discharge_data = df_d[required_cols_all].reset_index(drop=True)

            if not discharge_data.empty and len(discharge_data) > 2:
                all_integrated_data.append(discharge_data)
                print(f"✓ {battery_id}: {len(discharge_data)} cycles, 데이터 통합 완료")
            else:
                raise ValueError("로드된 데이터가 비어있거나 너무 적습니다.")

        except Exception as e:
            # 오류 발생 시에도 건너뛰고 계속 진행하도록 변경 (최대한 많은 데이터 확보)
            print(f"✗ {battery_id} 처리 실패 (건너뜀): {e}")
            continue

    if not all_integrated_data: 
        print("\n❌ 통합할 수 있는 유효한 데이터가 전혀 없습니다. 경로와 파일을 다시 확인해주세요.")
        return False
    
    df_final = pd.concat(all_integrated_data, ignore_index=True)
    df_final.to_csv(FILE_COMPLETE_DATA, index=False)
    print(f"\n✅ Step 1 완료. 저장: {FILE_COMPLETE_DATA}")
    return True
# ============================================================================
# 1.5. Step 1.5: 시간 데이터 후처리 (Time Post-processing)
# ============================================================================

def step1_post_process_time():
    print("\n" + "=" * 20 + " Step 1.5: 시간 데이터 후처리 " + "=" * 20)
    if not FILE_COMPLETE_DATA.exists(): return False

    df = pd.read_csv(FILE_COMPLETE_DATA)

    # 1. 'time_seconds'를 datetime 객체로 변환
    df['time_seconds'] = pd.to_datetime(df['time_seconds'])
    
    # 2. datetime을 Unix Epoch time (초)으로 변환하여 숫자형으로 만듭니다.
    # 이 과정이 Step 7의 .diff() 연산이 가능하게 합니다.
    df['time_seconds'] = df['time_seconds'].apply(lambda x: x.timestamp())

    # 3. Step 7을 위해 필요한 온도 컬럼의 결측치 처리 (Step 1에서 NaN으로 채웠을 경우)
    # 온도 데이터가 없어서 NaN으로 채워졌을 경우, Step 7의 temp_rise 계산을 위해 0으로 대체합니다.
    df['temperature_measured_max'] = df['temperature_measured_max'].fillna(df['temperature_measured_max'].median()).fillna(0)
    df['ambient_temp_c'] = df['ambient_temp_c'].fillna(df['ambient_temp_c'].median()).fillna(0)
    
    # 수정된 파일을 다음 단계 입력 파일로 덮어씁니다.
    df.to_csv(FILE_COMPLETE_DATA, index=False)
    print(f"✅ Step 1.5 완료. time_seconds를 Unix Timestamp로 변환 완료.")
    return True
# ============================================================================
# 2. Step 2: 임피던스 EIS 기반 외삽 (pre_2.py 기능)
# ============================================================================

def step2_eis_extrapolation():
    print("\n" + "=" * 25 + " Step 2: EIS 기반 외삽 " + "=" * 30)
    if not FILE_COMPLETE_DATA.exists(): return False

    df = pd.read_csv(FILE_COMPLETE_DATA)
    fields = ['re_ohm_interp', 'rct_ohm_interp', 'battery_impedance_mean_ohm_interp']
    
    for battery in df['battery'].unique():
        mask = df['battery'] == battery
        battery_df = df[mask].copy()

        # Re (직렬 저항): Forward Fill + 선형 외삽
        col = 're_ohm_interp'
        y = df.loc[mask, col].values
        valid_mask = ~np.isnan(y)

        if valid_mask.sum() > 0:
            first_valid_idx = np.where(valid_mask)[0][0]
            first_value = y[first_valid_idx]
            for i in range(first_valid_idx): y[i] = first_value

            if first_valid_idx < len(y) - 1:
                x = np.arange(len(y))
                x_valid, y_valid = x[valid_mask], y[valid_mask]
                f = interp1d(x_valid, y_valid, kind='linear', fill_value='extrapolate', bounds_error=False)
                y_filled = f(x)
                y[np.isnan(y)] = y_filled[np.isnan(y)]
            df.loc[mask, col] = y

        # Rct (전하전달 저항): 2차 다항식 외삽
        col = 'rct_ohm_interp'
        y = df.loc[mask, col].values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() >= 3:
            x_valid, y_valid = np.arange(len(y))[valid_mask], y[valid_mask]
            try:
                coeffs = np.polyfit(x_valid, np.maximum(y_valid, 1e-6), deg=2)
                poly = np.poly1d(coeffs)
                y_filled = np.maximum(poly(np.arange(len(y))), 0)
                y[np.isnan(y)] = y_filled[np.isnan(y)]
                df.loc[mask, col] = y
            except:
                pass # 실패 시 스킵

        # 나머지 임피던스 (선형 외삽)
        for col in ['battery_impedance_mean_ohm_interp']:
            y = df.loc[mask, col].values
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() >= 2:
                x_valid, y_valid = np.arange(len(y))[valid_mask], y[valid_mask]
                f = interp1d(x_valid, y_valid, kind='linear', fill_value='extrapolate', bounds_error=False)
                y_filled = np.maximum(f(np.arange(len(y))), 0)
                y[np.isnan(y)] = y_filled[np.isnan(y)]
                df.loc[mask, col] = y
    
    # 열 이름 통일 (pre_3.py 호환을 위해)
    df = df.rename(columns={'battery_impedance_mean_ohm_interp': 'battery_impedance_interp'})
    
    df.to_csv(FILE_COMPLETE_FINAL, index=False)
    print(f"✅ Step 2 완료. 저장: {FILE_COMPLETE_FINAL}")
    return True

# ============================================================================
# 3. Step 3: 열화 지표 계산 & K-Means 클러스터링 (pre_3.py 기능)
# ============================================================================

def step3_metrics_and_clustering():
    print("\n" + "=" * 25 + " Step 3: 지표 계산 및 클러스터링 " + "=" * 20)
    if not FILE_COMPLETE_FINAL.exists(): return False

    df = pd.read_csv(FILE_COMPLETE_FINAL)
    metrics = []

    for bid in sorted(df['battery'].unique()):
        data = df[df['battery'] == bid].sort_values('cycle_num')
        if len(data) < 5 or np.any(np.isnan(data['capacity_ahr'].values)): continue
        
        capacity = data['capacity_ahr'].values
        re_ohm = data['re_ohm_interp'].values
        impedance = data['battery_impedance_interp'].values
        voltage = data['voltage_measured_mean'].values

        regen_idx = np.argmax(capacity)
        capacity_after_regen = capacity[regen_idx]
        cap_deg = (capacity_after_regen - capacity[-1]) / capacity_after_regen * 100

        if abs(cap_deg) > 200: continue

        # DCR 성장률
        dcr_growth = ((re_ohm[-1] - re_ohm[0]) / abs(re_ohm[0]) * 100) if abs(re_ohm[0]) > 0.001 else 0
        dcr_growth = np.clip(dcr_growth, -100, 100)
        
        # 임피던스 성장률
        impedance_growth = ((impedance[-1] - impedance[0]) / abs(impedance[0]) * 100) if abs(impedance[0]) > 0.001 else 0
        impedance_growth = np.clip(impedance_growth, -100, 100)

        voltage_deg = voltage[0] - voltage[-1]
        lli_indicator = voltage_deg / abs(cap_deg) if abs(cap_deg) > 0.1 else 0
        lam_indicator = dcr_growth / abs(cap_deg) if abs(cap_deg) > 0.1 else 0

        metrics.append({'battery': bid, 'regen_cycles': data['cycle_num'].iloc[regen_idx], 'capacity_after_regen': capacity_after_regen, 'capacity_deg': cap_deg, 'dcr_growth': dcr_growth, 'impedance_growth': impedance_growth, 'lli_indicator': lli_indicator, 'lam_indicator': lam_indicator})

    metrics_df = pd.DataFrame(metrics).replace([np.inf, -np.inf], np.nan).dropna()
    
    # K-Means 클러스터링
    features_array = metrics_df[['capacity_deg', 'dcr_growth', 'impedance_growth', 'lli_indicator', 'lam_indicator']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    metrics_df['cluster'] = kmeans.fit_predict(features_scaled)
    
    # 원본 데이터에 매핑
    df_final = df.copy()
    for col in ['regen_cycles', 'capacity_after_regen', 'capacity_deg', 'cluster']:
        df_final[col] = df_final['battery'].map(metrics_df.set_index('battery')[col].to_dict())
    
    df_final.to_csv(FILE_FINAL_TRAINING, index=False)
    print(f"✅ Step 3 완료. 저장: {FILE_FINAL_TRAINING}")
    return True

# ============================================================================
# 4. Step 4: 데이터 정제 (pre_4.py 기능)
# ============================================================================

def step4_data_cleaning():
    print("\n" + "=" * 25 + " Step 4: 데이터 정제 " + "=" * 30)
    if not FILE_FINAL_TRAINING.exists(): return False

    df = pd.read_csv(FILE_FINAL_TRAINING)
    exclude_batteries = ['B0050', 'B0052']
    df_clean = df[~df['battery'].isin(exclude_batteries)].copy()
    
    df_clean.to_csv(FILE_CLEANED_V1, index=False)
    print(f"✅ Step 4 완료 (B0050, B0052 제거). 저장: {FILE_CLEANED_V1} (1차)")
    return True

# ============================================================================
# 5. Step 5: DCR, LLi, LAM 추가 (pre_7.py 기능)
# ============================================================================

def step5_add_lli_lam_dcr():
    print("\n" + "=" * 25 + " Step 5: LLi, LAM, DCR 추가 " + "=" * 25)
    if not FILE_CLEANED_V1.exists(): return False

    df = pd.read_csv(FILE_CLEANED_V1)
    
    for bid in sorted(df['battery'].unique()):
        data = df[df['battery'] == bid].sort_values('cycle_num')
        capacity = data['capacity_ahr'].values
        re_ohm = data['re_ohm_interp'].values
        voltage = data['voltage_measured_mean'].values
        
        regen_idx = np.argmax(capacity)
        cap_after_regen = capacity[regen_idx]
        cap_deg = (cap_after_regen - capacity[-1]) / cap_after_regen * 100 # pre_3와 동일한 cap_deg를 사용
        
        # DCR
        dcr_change = ((re_ohm[-1] - re_ohm[0]) / abs(re_ohm[0]) * 100) if abs(re_ohm[0]) > 0.001 else 0
        dcr_change = np.clip(dcr_change, -100, 100)
        
        # LLi
        voltage_deg = voltage[0] - voltage[-1]
        lli = np.clip(voltage_deg / abs(cap_deg), -100, 100) if abs(cap_deg) > 0.1 else 0

        # LAM
        lam = np.clip(dcr_change / abs(cap_deg), -100, 100) if abs(cap_deg) > 0.1 else 0

        df.loc[df['battery'] == bid, 'dcr'] = dcr_change
        df.loc[df['battery'] == bid, 'lli'] = lli
        df.loc[df['battery'] == bid, 'lam'] = lam
    
    df.to_csv(FILE_CLEANED_V1, index=False) # 2차 저장
    print(f"✅ Step 5 완료. 저장: {FILE_CLEANED_V1} (2차)")
    return True

# ============================================================================
# 6. Step 6: EOL 정보 계산 (pre_5.py 기능)
# ============================================================================

def step6_calculate_eol():
    print("\n" + "=" * 25 + " Step 6: EOL 정보 계산 " + "=" * 28)
    if not FILE_CLEANED_V1.exists(): return False

    df = pd.read_csv(FILE_CLEANED_V1)
    eol_data = []

    for battery in sorted(df['battery'].unique()):
        batt_data = df[df['battery'] == battery].sort_values('cycle_num')
        capacity_after_regen = batt_data['capacity_after_regen'].iloc[0]
        regen_cycles = int(batt_data['regen_cycles'].iloc[0])

        eol_threshold = capacity_after_regen * 0.7 # 30% 감소
        
        below_threshold = batt_data[batt_data['capacity_ahr'] <= eol_threshold]
        eol_dcycle = int(below_threshold['cycle_num'].iloc[0]) if len(below_threshold) > 0 else int(batt_data['cycle_num'].max())

        eol_data.append({'battery': battery, 'capacity_after_regen': capacity_after_regen, 'eol_threshold': eol_threshold, 'eol_dcycle': eol_dcycle, 'regen_cycles': regen_cycles, 'cluster': int(batt_data['cluster'].iloc[0])})

    eol_df = pd.DataFrame(eol_data)
    eol_df.to_csv(FILE_EOL_INFO, index=False)
    print(f"✅ Step 6 완료. 저장: {FILE_EOL_INFO}")
    return True

# ============================================================================
# 7. Step 7: 광범위한 피처 엔지니어링 (pre_10.py 기능)
# ============================================================================

def step7_feature_engineering():
    print("\n" + "=" * 25 + " Step 7: 광범위 피처 엔지니어링 " + "=" * 20)
    if not FILE_CLEANED_V1.exists(): return False

    df = pd.read_csv(FILE_CLEANED_V1)
    
    # 1️⃣ 용량 관련
    df['soh'] = df['capacity_ahr'] / df['capacity_after_regen']
    df['capacity_norm'] = df['capacity_ahr'] / df['capacity_after_regen']
    df['capacity_derivative'] = df.groupby('battery')['capacity_ahr'].diff()
    df['regen_strength'] = df['capacity_after_regen'] - df['capacity_ahr']

    # 2️⃣ 임피던스 관련
    df['impedance_sum'] = df['re_ohm_interp'] + df['rct_ohm_interp']
    initial_impedance = df.groupby('battery')['impedance_sum'].transform('first')
    df['impedance_growth'] = ((df['impedance_sum'] - initial_impedance) / (initial_impedance + 1e-9) * 100).fillna(0)
    initial_dcr = df.groupby('battery')['dcr'].transform('first')
    df['dcr_growth'] = ((df['dcr'] - initial_dcr) / (initial_dcr + 1e-9) * 100).fillna(0)

    # 3️⃣ 열화 지표 스케일링 & 평활
    for col in ['lli', 'lam', 'dcr']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_scaled'] = np.clip((df[col] - mean) / (std + 1e-9), -3, 3)
        df[f'{col}_smooth'] = df.groupby('battery')[col].transform(lambda x: x.rolling(window=5, center=True, min_periods=1).mean())

    # 4️⃣ 스트레스 / 열 관련 (컬럼 존재 가정)
    df['temp_rise'] = df['temperature_measured_max'] - df['ambient_temp_c']
    df['thermal_stress'] = df['temp_rise'] / (df['discharge_time_sec'] + 1e-9)
    df['ir_drop'] = df['voltage_load_mean'] - df['voltage_measured_mean']
    df['voltage_range'] = df['voltage_measured_max'] - df['voltage_measured_min']
    df['C_rate_avg'] = df['current_measured_mean'] / (df['capacity_after_regen'] + 1e-9)
    df['C_rate_max'] = (df['current_measured_mean'] + 2 * df['current_measured_std']) / (df['capacity_after_regen'] + 1e-9)
    df['current_temp_product'] = df['current_measured_mean'] * df['temp_rise']
    df['load_temp_interact'] = df['C_rate_avg'] * df['temp_rise']

    # 5️⃣ 시간/사이클 파생
    df['cycle_diff_time'] = df.groupby('battery')['time_seconds'].diff()
    df['cycle_duration_rate'] = df['cycle_diff_time'] / (df['discharge_time_sec'] + 1e-9)

    # 6️⃣ 인코딩 & 구간화
    le = LabelEncoder()
    df['battery_encoded'] = le.fit_transform(df['battery'])
    df['_crate_bin'] = pd.cut(df['C_rate_avg'], bins=5, labels=False, duplicates='drop')
    df['_temp_bin'] = pd.cut(df['temp_rise'], bins=5, labels=False, duplicates='drop')
    
    df.to_csv(FILE_COMPLETED, index=False)
    print(f"✅ Step 7 완료. 저장: {FILE_COMPLETED}")
    return True

# ============================================================================
# 8. Step 8: 최종 정제 및 학습 데이터 완성 (pre_9.py 기능)
# ============================================================================

def step8_final_cleaning():
    print("\n" + "=" * 25 + " Step 8: 최종 정제 및 학습 데이터 완성 " + "=" * 14)
    if not FILE_COMPLETED.exists(): return False

    df = pd.read_csv(FILE_COMPLETED)
    
    # 1. ALL-NaN 컬럼 동적 제거
    # Step 1에서 NaN으로 채워진 NASA 원본 컬럼과 Step 7에서 NaN 기반으로 생성된 파생 피처를 모두 제거합니다.
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols, errors='ignore')
        print(f"   (디버그) ALL NaN 컬럼 제거 (NaN 컬럼 전체): {all_nan_cols}")

    # 2. capacity_ahr = 0 제거
    df = df[df['capacity_ahr'] > 0].copy()

    # 3. 결측치 처리 (배터리별 중앙값으로 fillna)
    # capacity_derivative, cycle_diff_time 등은 첫 번째 사이클에서 NaN이 됩니다.
    for col in ['capacity_derivative', 'cycle_diff_time', 'cycle_duration_rate']:
        df[col] = df.groupby('battery')[col].transform(lambda x: x.fillna(x.median()))
    
    # 4. ⚠️ 핵심 수정: 나머지 모든 결측치(NaN)는 0으로 대체 (Imputation)
    # 이로써 행 전체를 삭제하는 것을 방지하고, 유효한 배터리 데이터를 보존합니다.
    df = df.fillna(0)
    
    # 5. 극단값 처리
    df['dcr_growth'] = np.clip(df['dcr_growth'], -1000, 1000)
    for col in ['impedance_growth', 'impedance_sum']:
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = np.clip(df[col], q1, q99)

    # 6. 음수값 처리 (절댓값 생성)
    for col in ['C_rate_avg', 'C_rate_max', 'current_temp_product', 'load_temp_interact', 'ir_drop']:
        if col in df.columns:
            df[f'{col}_abs'] = np.abs(df[col])
        
    # 7. 로그 변환 (선택적)
    if 'impedance_growth' in df.columns:
        df['impedance_growth_log'] = np.sign(df['impedance_growth']) * np.log(1 + np.abs(df['impedance_growth']))
    if 'dcr_growth' in df.columns:
        df['dcr_growth_log'] = np.sign(df['dcr_growth']) * np.log(1 + np.abs(df['dcr_growth']))

    # 8. 불필요한 컬럼 제거
    if 'capacity_norm' in df.columns:
        df = df.drop('capacity_norm', axis=1)

    # 최종적으로 유효한 데이터가 남아 있는지 확인
    if df.empty or df['battery'].nunique() == 0:
        print("❌ Step 8 완료 후, 유효한 학습 데이터가 남아있지 않아 Step 9를 건너뜁니다.")
        return False

    df.to_csv(FILE_CLEANED_FINAL, index=False)
    print(f"✅ Step 8 완료. 저장: {FILE_CLEANED_FINAL} (목표 1 달성)")
    print(f"   (디버그) 최종 학습 데이터 배터리 수: {df['battery'].nunique()}개, 총 행 수: {len(df)}")
    return True
# ============================================================================
# 9. Step 9: Advanced CEEMDAN 분해 (ceemdan_3.py 기능)
# ============================================================================

def step9_ceemdan_decomposition():
    print("\n" + "=" * 25 + " Step 9: Advanced CEEMDAN 분해 " + "=" * 22)
    if not FILE_CLEANED_FINAL.exists(): return False

    df = pd.read_csv(FILE_CLEANED_FINAL)
    ceemdan = AdvancedCEEMDAN(noise_mode='adaptive', n_ensembles=150, seed=42)
    
    results = []
    batteries = sorted(df['battery'].unique())
    
    print(f"🔄 Capacity 신호 분해 중... ({len(batteries)} 배터리)")
    
    for bat_idx, battery_id in enumerate(batteries):
        battery_data = df[df['battery'] == battery_id].sort_values('cycle_num')
        signal = battery_data['capacity_ahr'].values
        cycle_nums = battery_data['cycle_num'].values
        
        if len(signal) < 5: continue

        imfs, residual = ceemdan.decompose(signal, max_imfs=6)
        
        # 결과 저장
        for i, cycle_num in enumerate(cycle_nums):
            result_row = {'battery': battery_id, 'cycle_num': cycle_num, 'Capacity': signal[i], 'Residual': residual[i]}
            for j in range(6): result_row[f'IMF{j+1}'] = imfs[j][i]
            results.append(result_row)
        
        print(f"  [{bat_idx+1:2d}/{len(batteries)}] {battery_id} 분해 완료", end='\r', flush=True)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(FILE_CEEMDAN_FINAL, index=False)
    print(f"\n✅ Step 9 완료. 저장: {FILE_CEEMDAN_FINAL} (목표 2 달성)")
    return True

# ============================================================================
# 최종 실행
# ============================================================================

def main_pipeline():
    # 1. 원시 데이터 처리
    if not step1_extract_and_interpolate(): return
    # 1.5. 시간 데이터 후처리 (새로 추가)
    if not step1_post_process_time(): return
    # 2. 임피던스 외삽
    if not step2_eis_extrapolation(): return
    # 3. 클러스터링
    if not step3_metrics_and_clustering(): return
    # 4. 정제
    if not step4_data_cleaning(): return
    # 5. LLi, LAM, DCR 추가
    if not step5_add_lli_lam_dcr(): return
    # 6. EOL 계산 (이 단계는 출력 파일만 생성하며, 아래 단계에 직접적인 입력이 아님. 필수는 아니나 워크플로우를 위해 실행)
    step6_calculate_eol() 
    # 7. 피처 엔지니어링
    if not step7_feature_engineering(): return
    # 8. 최종 정제 및 학습 데이터 완성 (목표 1 생성)
    if not step8_final_cleaning(): return
    # 9. CEEMDAN 분해 (목표 2 생성)
    if not step9_ceemdan_decomposition(): return
    
    print("\n" + "=" * 80)
    print("✨ 모든 파이프라인 작업이 완료되었습니다!")
    print(f"최종 학습 데이터셋: {FILE_CLEANED_FINAL}")
    print(f"CEEMDAN 분해 데이터셋: {FILE_CEEMDAN_FINAL}")
    print("=" * 80)

if __name__ == "__main__":
    main_pipeline()