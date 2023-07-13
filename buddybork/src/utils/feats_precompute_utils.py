import os
from typing import List

from ossr_utils.io_utils import save_pickle

from src.utils.db_utils import execute_pure_sql
from src.utils.feats_io_utils import get_feat_cache_fname, extract_features_one, get_seg_df_for_days


def extract_feats_for_annots_all(verbose: int = 0):
    sql = "SELECT * FROM annots ORDER BY datetime_start DESC"
    df = execute_pure_sql(sql)
    if verbose >= 1:
        print('Precomputing feats for annotated segments ({})'.format(len(df)))
    if verbose >= 3:
        print(df)

    for i, row in df.iterrows():
        dt_start = row['datetime_start']
        dt_end = row['datetime_end']
        tag = row['tag']

        cache_fname = get_feat_cache_fname(dt_start, dt_end, tag)
        if not os.path.exists(cache_fname):
            if verbose >= 2:
                print('Extracting {}/{} features for {} ({}, {})'.format(i + 1, len(df), tag, dt_start, dt_end))
                print('  saving to ' + cache_fname)
            X_i, meta_i = extract_features_one(dt_start, dt_end=dt_end, tag=tag)[1:]
            save_pickle(cache_fname, dict(X=X_i, meta=meta_i), make_dir=True)


def extract_feats_for_segs_all(days: List[str],
                               verbose: int = 0):
    """Extract features for all raw segs"""
    for day in days:
        df = get_seg_df_for_days([day])
        if len(df) == 0:
            continue
        if verbose >= 1:
            print('Precomputing feats for ' + day + ' ({})'.format(len(df)))
        if verbose >= 3:
            print(day)
            print(df)

        for i, row in df.iterrows():
            dt_ = row['datetime']

            cache_fname = get_feat_cache_fname(dt_)
            # print(cache_fname)
            if not os.path.exists(cache_fname):
                if verbose >= 2:
                    print('Extracting {}/{} features ({})'.format(i + 1, len(df), dt_))
                    print('  saving to ' + cache_fname)
                X_i, meta_i = extract_features_one(dt_)[1:]
                save_pickle(cache_fname, dict(X=X_i, meta=meta_i), make_dir=True)
