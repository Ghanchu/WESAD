import pandas as pd
import numpy as np
import pickle
import glob
import os
import time
from format_and_process import format_and_process_data

pkl_paths = sorted(glob.glob(os.path.join('Data', 'S*', 'S*.pkl')))
n_subjects = len(pkl_paths)
print(f'Found {n_subjects} participant files:')
for p in pkl_paths:
    print(f'  {p}')

full_data = pd.DataFrame(columns=[
    'Participant ID', 'Window ID',
    'ECG_HR_mean', 'ECG_HR_std', 'ECG_HRV_mean', 'ECG_HRV_std', 'ECG_NN50', 'ECG_pNN50', 'ECG_TINN',
    'ECG_rmsHRV', 'ECG_HRV_ULF', 'ECG_HRV_LF', 'ECG_HRV_HF', 'ECG_HRV_UHF', 'ECG_HRV_LF_HF_ratio',
    'ECG_HRV_sum_ULF_HF', 'ECG_HRV_rel_power', 'ECG_HRV_LFnorm', 'ECG_HRV_HFnorm',
    'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_slope', 'EDA_range',
    'EDA_SCL_mean', 'EDA_SCL_std', 'EDA_SCR_std', 'EDA_SCL_corr',
    'EDA_SCR_count', 'EDA_SCR_magnitude_sum', 'EDA_SCR_duration_sum', 'EDA_SCR_area',
    'EMG_mean', 'EMG_std', 'EMG_range', 'EMG_integral', 'EMG_median', 'EMG_p10', 'EMG_p90',
    'EMG_freq_mean', 'EMG_freq_median', 'EMG_freq_peak',
    'EMG_PSD_band1', 'EMG_PSD_band2', 'EMG_PSD_band3', 'EMG_PSD_band4',
    'EMG_PSD_band5', 'EMG_PSD_band6', 'EMG_PSD_band7',
    'EMG_peaks_count', 'EMG_peak_amp_mean', 'EMG_peak_amp_std',
    'EMG_peak_amp_sum', 'EMG_peak_amp_norm_sum',
    'RESP_inhal_mean', 'RESP_inhal_std', 'RESP_exhal_mean', 'RESP_exhal_std', 'RESP_IE_ratio',
    'RESP_stretch', 'RESP_vol_insp', 'RESP_rate', 'RESP_duration',
    'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max', 'TEMP_range', 'TEMP_slope',
    'Label',
])

subject_dfs = []
pipeline_start = time.time()

for idx, path in enumerate(pkl_paths, 1):
    subject_id = os.path.basename(path).replace('.pkl', '')
    print(f'\n{"="*55}')
    print(f'  Subject {idx}/{n_subjects}: {subject_id}')
    print(f'{"="*55}')

    t0 = time.time()
    print(f'  Loading {path} ...')
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(f'  Loaded in {time.time()-t0:.1f}s')

    df = format_and_process_data(data, subject_id)
    subject_dfs.append(df)

    cumulative = sum(len(d) for d in subject_dfs)
    elapsed    = time.time() - pipeline_start
    remaining  = n_subjects - idx
    avg_per    = elapsed / idx
    eta        = avg_per * remaining
    print(f'  Cumulative windows so far: {cumulative:,}  |  '
          f'Elapsed: {elapsed:.0f}s  |  ETA: {eta:.0f}s')

print(f'\n{"="*55}')
print('  Concatenating all subjects ...')
full_data = pd.concat([full_data] + subject_dfs, ignore_index=True)

total_time = time.time() - pipeline_start
print(f'\nAll done in {total_time:.0f}s.')
print(f'Total windows : {len(full_data):,}')
print(full_data['Label'].value_counts().rename({0: 'non-stress', 1: 'stress'}))

out_path = 'full_data.csv'
print(f'\nSaving to {out_path} ...')
full_data.to_csv(out_path, index=False)
print(f'Saved.')
