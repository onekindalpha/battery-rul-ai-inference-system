# 기존 import는 유지

# 설정 (INPUT_DIR는 .xlsx 파일이 있는 디렉토리를 가리켜야 함)
INPUT_DIR = Path('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/ML/raw')
OUTPUT_DIR = Path('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# 💡 참고: 엑셀 파일에는 MATLAB date vector가 없을 수 있으므로 이 함수는 사용되지 않거나 수정되어야 함
# 엑셀 파일에 'Time_Seconds' 또는 'Timestamp' 열이 있다고 가정합니다.
def matlab_datevector_to_seconds(datevec):
    """(엑셀용) 날짜/시간 포맷에 따라 수정 필요. 여기서는 기존 함수를 유지하되 사용하지 않음."""
    pass

def safe_float(val):
    """배열 또는 스칼라를 안전하게 float로 변환"""
    # 엑셀 데이터는 이미 스칼라일 가능성이 높으므로 단순화
    try:
        if isinstance(val, (np.ndarray, pd.Series)):
            return float(val.iloc[0]) if val.size > 0 else np.nan
        return float(val)
    except:
        return np.nan

def extract_all_battery_data_xlsx(excel_file):
    """배터리 .xlsx 파일에서 모든 discharge + impedance 추출 (엑셀 구조 가정)"""
    battery_id = excel_file.stem
    
    # ⚠️ 중요: 엑셀 파일이 시트를 어떻게 구성했는지에 따라 sheet_name 인수를 조정해야 합니다.
    try:
        # 모든 시트를 딕셔너리로 로드합니다.
        # keys는 시트 이름(예: 'Discharge', 'Impedance'), values는 DataFrame입니다.
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        
        discharge_data = pd.DataFrame()
        impedance_data = pd.DataFrame()

        # === STEP 1: DISCHARGE 데이터 추출 (시트 이름 'Discharge' 가정) ===
        if 'Discharge' in excel_data:
            df_d = excel_data['Discharge'].copy()
            df_d['battery'] = battery_id
            
            # 💡 열 이름 매핑: 엑셀 파일의 실제 열 이름으로 대체해야 합니다.
            # 예시 매핑: Excel_Column_Name -> Target_Column_Name
            df_d = df_d.rename(columns={
                'Cycle_Number': 'cycle_num',
                'Time_Seconds': 'time_seconds',
                'Capacity_Ah': 'capacity_ahr',
                'Ambient_Temperature_C': 'ambient_temp_c',
                'Voltage_Mean': 'voltage_measured_mean',
                'Temperature_Mean': 'temperature_measured_mean',
                'Discharge_Duration': 'discharge_time_sec'
            })
            
            # 필요한 열만 선택하고, NaN 및 기타 정리 로직 추가
            # ⚠️ 사용자의 실제 엑셀 열 이름에 맞춰 목록을 수정하세요.
            required_cols = [
                'battery', 'cycle_num', 'time_seconds', 'capacity_ahr',
                'ambient_temp_c', 'voltage_measured_mean', 'temperature_measured_mean',
                'discharge_time_sec'
            ]
            
            # 없는 열은 NaN으로 채워서 강제로 포함
            for col in required_cols:
                if col not in df_d.columns:
                    df_d[col] = np.nan
            
            discharge_data = df_d[required_cols].reset_index(names='cycle_index')
            discharge_data['cycle_index'] = discharge_data.index # 인덱스 재설정
            
            # .mat 파일에서 추출했던 통계 열도 계산해야 한다면 여기에 추가
            # 예: df_d['current_measured_mean'] = df_d['Current_Measured'].mean()

        # === STEP 2: IMPEDANCE 데이터 추출 (시트 이름 'Impedance' 가정) ===
        if 'Impedance' in excel_data:
            df_i = excel_data['Impedance'].copy()
            df_i['battery'] = battery_id
            
            # 💡 열 이름 매핑: 엑셀 파일의 실제 열 이름으로 대체해야 합니다.
            df_i = df_i.rename(columns={
                'Time_Seconds': 'time_seconds',
                'Re_Ohm': 're_ohm',
                'Rct_Ohm': 'rct_ohm',
                'Battery_Impedance_Mean': 'battery_impedance_mean_ohm',
                # 다른 임피던스 관련 열도 여기에 매핑
            })
            
            required_cols_imp = [
                'battery', 'time_seconds', 're_ohm', 'rct_ohm',
                'battery_impedance_mean_ohm'
            ]
            
            # 없는 열은 NaN으로 채워서 강제로 포함
            for col in required_cols_imp:
                if col not in df_i.columns:
                    df_i[col] = np.nan
            
            impedance_data = df_i[required_cols_imp].reset_index(names='cycle_index')
            
            # 복소수 처리 (엑셀 데이터는 보통 실수이므로 간단하게 처리)
            impedance_data['re_ohm'] = impedance_data['re_ohm'].apply(safe_float)
            impedance_data['rct_ohm'] = impedance_data['rct_ohm'].apply(safe_float)


        print(f"✓ {battery_id}: {len(discharge_data)} discharge, {len(impedance_data)} impedance (from XLSX)")
        return discharge_data, impedance_data

    except Exception as e:
        print(f"✗ {battery_id} 처리 실패: {e}")
        return pd.DataFrame(), pd.DataFrame()


# === MAIN 함수 수정 ===
def main_xlsx():
    print("=" * 80)
    print("배터리 완전 분석 - XLSX 파일 처리 및 Cubic Spline 보간")
    print("=" * 80)

    # .mat 대신 .xlsx 파일을 찾습니다.
    excel_files = sorted(INPUT_DIR.glob('B*.xlsx')) # 🔍 확장자 변경

    if not excel_files:
        print("❌ .xlsx 파일을 찾을 수 없습니다!")
        return

    print(f"\n발견된 파일: {len(excel_files)}개\n")

    all_integrated_data = []

    # 각 배터리 처리
    for excel_file in excel_files: # 🔄 파일 변수 이름 변경
        # 💡 수정된 엑셀 로드 함수 사용
        discharge_df, impedance_df = extract_all_battery_data_xlsx(excel_file) 

        if not discharge_df.empty:
            # Cubic Spline 보간은 시간 열(`time_seconds`)이 있다면 동일하게 적용 가능합니다.
            discharge_df = interpolate_impedance_cubic(discharge_df, impedance_df)
            all_integrated_data.append(discharge_df)

    # (이하 코드는 동일합니다: concat, 저장, 통계 출력)

    if not all_integrated_data:
        print("\n❌ 추출된 데이터가 없습니다!")
        return

    # 통합
    final_df = pd.concat(all_integrated_data, ignore_index=True)

    print(f"\n" + "=" * 80)
    print(f"✓ 총 Discharge cycles: {len(final_df)}")
    print(f"✓ 배터리 수: {final_df['battery'].nunique()}")
    print("=" * 80)

    # 단일 통합 CSV로 저장
    output_file = OUTPUT_DIR / 'battery_complete_data_xlsx.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n✓ 저장: {output_file}")
    print(f"  - 행 수: {len(final_df)}")
    print(f"  - 열 수: {len(final_df.columns)}")

    # 통계
    print(f"\n데이터 통계:")
    stats_cols = ['cycle_num', 'capacity_ahr', 'voltage_measured_mean',
                  'temperature_measured_mean', 're_ohm_interp', 'rct_ohm_interp',
                  'battery_impedance_interp', 'rectified_impedance_interp']
    print(final_df[stats_cols].describe().round(4))

    # 배터리별 요약
    print(f"\n배터리별 요약:")
    battery_summary = final_df.groupby('battery').agg({
        'cycle_num': 'max',
        'capacity_ahr': ['min', 'max', 'mean'],
        'ambient_temp_c': 'first',
        're_ohm_interp': ['count', 'mean', 'max']
    }).round(4)
    print(battery_summary)

    return final_df


if __name__ == "__main__":
    df = main_xlsx() # 💡 메인 함수 호출 변경

    # (샘플 데이터 출력 로직은 기존과 동일)
    if df is not None and not df.empty:
        print("\n" + "=" * 80)
        print("샘플 데이터 (각 배터리별 처음 2행):")
        print("=" * 80)

        sample_cols = ['battery', 'cycle_num', 'capacity_ahr', 'ambient_temp_c',
                       'voltage_measured_mean', 'temperature_measured_mean',
                       're_ohm_interp', 'rct_ohm_interp',
                       'battery_impedance_interp', 'rectified_impedance_interp']

        for battery_id in df['battery'].unique()[:5]:  # 처음 5개 배터리
            print(f"\n{battery_id}:")
            bat_data = df[df['battery'] == battery_id]
            print(bat_data[sample_cols].head(2).to_string())