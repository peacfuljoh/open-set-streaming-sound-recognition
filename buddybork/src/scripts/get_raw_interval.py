"""Get interval of raw data segs"""

from datetime import datetime

import numpy as np

from src.utils.db_utils import execute_pure_sql
from src.constants_stream import BUFF_LEN, BUFF_DUR, SAMPLERATE

from ossr_utils.audio_utils import write_wav, read_wav

if 0:
    # borkfest
    name = 'borkfest'
    dt_start_s = '2023-01-29 19:31:37'
    dt_end_s = '2023-01-29 19:36:30'
if 1:
    # dreams
    name = 'dreams'
    dt_start_s = '2023-02-14 01:39:55'
    dt_end_s = '2023-02-14 01:40:40'

sql = "SELECT * FROM raw WHERE datetime BETWEEN '" + dt_start_s + "' AND '" + dt_end_s + "'"
df = execute_pure_sql(sql)

fmt = "%Y-%m-%d %H:%M:%S"
dt_start = datetime.strptime(dt_start_s, fmt)
dt_end = datetime.strptime(dt_end_s, fmt)
dur = (dt_end - dt_start).total_seconds()
num_buffs = dur / BUFF_DUR
num_samps = int((num_buffs + 5) * BUFF_LEN)

wf = np.zeros(num_samps, dtype='int16')
t_all = []
for i, row in df.iterrows():
    sr, wf_i = read_wav(row['filepath'])
    t_start = (row['datetime'] - dt_start).total_seconds() # seconds from start of rec
    t0 = int(t_start * SAMPLERATE)
    wf[t0:t0 + len(wf_i)] += wf_i
    t_all.append(t_start)

pad_dur = 5
tmin = max(0, int((np.min(t_all) - pad_dur) * SAMPLERATE))
tmax = min(len(wf), int((np.max(t_all) + BUFF_DUR + pad_dur) * SAMPLERATE))
wf = wf[tmin:tmax]

fpath = '/home/nuc/Desktop/' + name + '_' + dt_start_s[:10] + '.wav'
write_wav(fpath, sr, wf, verbose=True)
