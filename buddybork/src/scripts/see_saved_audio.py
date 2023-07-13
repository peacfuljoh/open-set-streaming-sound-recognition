
import os

from matplotlib import pyplot as plt
import numpy as np

from ossr_utils.audio_utils import read_wav
from src.constants_stream import DATA_DIR, SAMPLERATE, BUFF_DUR


sess_id = 1679455625147
sess_dir = os.path.join(DATA_DIR, str(sess_id))

fnames = [f for f in sorted(os.listdir(sess_dir)) if '.wav' in f]

if 1:
    ts_ = 1679509634971
    ts_min = ts_ - 60 * 10
    ts_max = ts_ + 60 * 10
    fnames = [f for f in fnames if (ts_min <= int(f[:-4]) <= ts_max)]

wfs = []
for fname in fnames:
    fpath = os.path.join(sess_dir, fname)
    sr, wf = read_wav(fpath)
    wf = wf / 2 ** 15
    wfs.append(wf)

print(len(wfs))

ts0 = 60.0 * 0
tsd = 60.0 * 20

wf_all = np.concatenate(wfs[int(ts0 / BUFF_DUR):int((ts0 + tsd) / BUFF_DUR)])
ts = np.arange(len(wf_all)) / SAMPLERATE + ts0

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ts, wf_all)
plt.show()

