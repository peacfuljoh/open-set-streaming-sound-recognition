"""Main process to pre-compute features on a regular basis"""

from typing import List
import time
import datetime
from threading import Thread

from src.utils.feats_precompute_utils import extract_feats_for_annots_all, extract_feats_for_segs_all
from src.utils.feats_io_utils import expand_days_list


FEAT_PRECOMP_DAYS = 7
PRECOMP_INTVL_SEGS = 30 # seconds
PRECOMP_INTVL_ANNOTS = 60 * 60 * 24 # seconds


def get_precomp_day_range() -> List[str]:
    dt_now = datetime.datetime.now()
    dt_start = (dt_now - datetime.timedelta(days=FEAT_PRECOMP_DAYS)).date()
    dt_end = dt_now.date()
    days = [str(dt_start), str(dt_end)]
    return days

def precomp_segs():
    while 1:
        day: str = get_precomp_day_range()[1]
        extract_feats_for_segs_all([day], verbose=2) # TODO: add time info to only scan through recent segs
        time.sleep(PRECOMP_INTVL_SEGS)

def precomp_annots():
    while 1:
        # days = get_precomp_day_range()
        # days = expand_days_list([days])
        extract_feats_for_annots_all(verbose=2)
        time.sleep(PRECOMP_INTVL_ANNOTS)



if __name__ == '__main__':
    seg_thread = Thread(target=precomp_segs, daemon=True)
    seg_thread.start()

    annot_thread = Thread(target=precomp_annots, daemon=True)
    annot_thread.start()

    while 1:
        time.sleep(60)
