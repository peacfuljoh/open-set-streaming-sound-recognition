"""Try out feature pre-caching functionalities"""

import time
import datetime

import numpy as np

from ossr_utils.io_utils import load_pickle

from src.utils.feats_io_utils import expand_days_list
from src.utils.feats_precompute_utils import extract_feats_for_annots_all, extract_feats_for_segs_all



DAYS_ALL = [['2022-11-01', '2024-01-01']]



def try_cache_load():
    """Test performance of annot vs seg feature file load"""
    N = 50

    # fname_seg = '/home/nuc/buddybork_features/2023-03-01s06c58c44p084000.pickle'
    # fname_seg = '/home/nuc/buddybork_features/2023-03-01s19c28c05p405000.pickle'
    fname_seg = '/home/nuc/buddybork_features/2023-03-01s08c26c37p109000.pickle'
    ts_all = []
    for _ in range(N):
        t0 = time.time()
        seg = load_pickle(fname_seg)
        ts_all.append(time.time() - t0)
    print(ts_all)
    print(np.mean(ts_all))
    print(np.std(ts_all))

    # fname_annot = '/home/nuc/buddybork_features/2023-02-22s04c45c06p732000_2023-02-22s04c45c07p606000_snuck.pickle'
    # fname_annot = '/home/nuc/buddybork_features/2023-02-22s14c58c13p015000_2023-02-22s14c58c15p311000_door_close.pickle'
    fname_annot = '/home/nuc/buddybork_features/2023-02-28s19c12c16p864000_2023-02-28s19c12c18p636000_door_close.pickle'
    ts_all = []
    for _ in range(N):
        t1 = time.time()
        seg_annot = load_pickle(fname_annot)
        ts_all.append(time.time() - t1)
    print(ts_all)
    print(np.mean(ts_all))
    print(np.std(ts_all))






if __name__ == "__main__":
    verbose = 1
    # extract all feats for annotated segs
    if 1:
        extract_feats_for_annots_all(verbose=verbose)

    # extract feats for all segs
    if 1:
        if 1:
            days = DAYS_ALL
        if 0:
            dt_now = datetime.datetime.now()
            dt_start = (dt_now - datetime.timedelta(days=7)).date()
            dt_end = dt_now.date()
            days = [[str(dt_start), str(dt_end)]]
        if 0:
            days = [['2023-02-20', '2023-03-05']]
            days = ['2023-02-28']
        days = expand_days_list(days)
        extract_feats_for_segs_all(days, verbose=verbose)

    # try loading cached feature file
    if 0:
        try_cache_load()
