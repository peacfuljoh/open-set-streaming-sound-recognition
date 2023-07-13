
import numpy as np

from src.utils.feats_io_utils import get_annots_df_for_days, extract_features_one


days_info = [['2020-01-01', '2024-01-01']]

df_annots = get_annots_df_for_days(days_info)

print(df_annots)

amps = []
for i, row in df_annots.iterrows():
    wf_ = extract_features_one(row['datetime_start'], row['datetime_end'])[0]
    amp_i = np.max(np.abs(wf_))
    amps.append(amp_i)

print(sorted(amps))
