

import os

import pandas as pd

from src.utils.db_utils import insert_or_update_db
from ossr_utils.misc_utils import get_seg_amp_metric, convert_utc_to_dt
from ossr_utils.audio_utils import read_wav
from src.constants_stream import DATA_DIR, LEN_DATA_SESS_DIR


sess_dirs = [d for d in os.listdir(DATA_DIR) if len(d) == LEN_DATA_SESS_DIR]
for dpath in [os.path.join(DATA_DIR, d) for d in sess_dirs]:
    print(dpath)
    fnames = [f for f in os.listdir(dpath) if '.wav' in f]
    for fname in fnames:
        fpath = os.path.join(dpath, fname)
        # sr, wf = read_wav(fpath)

        # amp = get_seg_amp_metric(wf)
        df = pd.DataFrame({
            "datetime": [convert_utc_to_dt(int(fname[:-4]) / 1e3)],
            # "max_amp": [amp],
            "filepath": fpath
        })
        insert_or_update_db('raw', df, 'update')
