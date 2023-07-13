

import os

from src.utils.db_utils import execute_pure_sql_no_return
from ossr_utils.misc_utils import get_seg_amp_metric, convert_utc_to_dt
from src.constants_stream import MAX_AMP_THRESH, DATA_DIR, LEN_DATA_SESS_DIR

from ossr_utils.audio_utils import read_wav


amp_thresh = MAX_AMP_THRESH


amps = []


sess_dirs = [d for d in sorted(os.listdir(DATA_DIR)) if len(d) == LEN_DATA_SESS_DIR]
for dpath in [os.path.join(DATA_DIR, d) for d in sess_dirs]:
    print(dpath)
    fnames = [f for f in sorted(os.listdir(dpath)) if '.wav' in f]
    for fname in fnames:
        fpath = os.path.join(dpath, fname)
        sr, wf = read_wav(fpath)

        amp = get_seg_amp_metric(wf)
        amps.append(amp)

        if amp < amp_thresh:
            if 1:
                # delete from db
                dt = convert_utc_to_dt(int(fname[:-4]) / 1e3)
                print('Removing segment: ' + fname)
                q = "DELETE FROM raw WHERE datetime = '" + str(dt) + "'"
                execute_pure_sql_no_return(q)

                # delete from data dir
                os.remove(fpath)



if 1:
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.scatter(range(len(amps)), amps, c='b')
    ax.plot([0, len(amps) - 1], [amp_thresh] * 2, c='r')

    ax2 = fig.add_subplot(122)
    plt.hist(amps, 100, [0, 2000])

    plt.show()
