import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned.csv')

print("="*80)
print("Adding dcr, lam, lli columns")
print("="*80)

for bid in sorted(df['battery'].unique()):
    data = df[df['battery'] == bid].sort_values('cycle_num')
    capacity = data['capacity_ahr'].values
    re_ohm = data['re_ohm_interp'].values
    voltage = data['voltage_measured_mean'].values

    # 배터리 재생 후 용량
    regen_idx = np.argmax(capacity)
    cap_after_regen = capacity[regen_idx]

    # 용량 감소
    cap_deg = (cap_after_regen - capacity[-1]) / cap_after_regen * 100

    # DCR
    if abs(re_ohm[0]) > 0.001:
        dcr = ((re_ohm[-1] - re_ohm[0]) / abs(re_ohm[0]) * 100)
    else:
        dcr = 0
    dcr = np.clip(dcr, -100, 100)

    # LLi
    voltage_deg = voltage[0] - voltage[-1]
    if abs(cap_deg) > 0.1:
        lli = np.clip(voltage_deg / abs(cap_deg), -100, 100)
    else:
        lli = 0

    # LAM
    if abs(cap_deg) > 0.1:
        lam = np.clip(dcr / abs(cap_deg), -100, 100)
    else:
        lam = 0

    # 할당
    df.loc[df['battery'] == bid, 'dcr'] = dcr
    df.loc[df['battery'] == bid, 'lli'] = lli
    df.loc[df['battery'] == bid, 'lam'] = lam

    print(f"{bid}: dcr={dcr:.2f}, lli={lli:.6f}, lam={lam:.2f}")

# 저장
df.to_csv('/content/drive/MyDrive/Colab Notebooks/SAMSUNG_AI/battery/battery_project/db/nasabattery/output/battery_training_data_cleaned.csv', index=False)

print(f"\nOK: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")