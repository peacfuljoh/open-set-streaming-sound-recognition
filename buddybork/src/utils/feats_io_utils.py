'''
Feature extraction for specified days, only annotated segments or selection of all segments.
'''

import os
from typing import Union, List, Tuple, Optional
import time

from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

from ossr_utils.misc_utils import print_flush, get_times_from_dts, get_seg_amp_metric
from src.ml.features import FeatureSet
try:
    from src.utils.db_utils import execute_pure_sql
    from src.ml.featurization import featurize
    from src.constants_stream import BUFF_DUR, SAMPLERATE, MAX_VAL_RAW_AUDIO

    TD_BUFF_DUR = timedelta(milliseconds=BUFF_DUR * 1e3)
except:
    print_flush('feats_io_utils.py -> Could not import some modules.')
from ossr_utils.audio_utils import read_wav

from src.constants_ml import OTHER_TAG, MAX_BG_SEGS_PER_DAY, FEAT_DIR, MAX_NUM_SPEC_FEATS, META_FILT_FACTOR_BG
from ossr_utils.io_utils import save_pickle, load_pickle




PRINT_DEBUG = False




FMT_DT = "%Y-%m-%d %H:%M:%S.%f"


def day_str_to_date(s: str) -> date:
    return datetime.strptime(s, '%Y-%m-%d').date()

def expand_date_range(day_lims: List[str]) -> List[str]:
    days = []
    dt = day_str_to_date(day_lims[0])
    dt_end = day_str_to_date(day_lims[1])
    while (dt <= dt_end):
        dt_str = dt.strftime("%Y-%m-%d")
        days.append(dt_str)
        dt += timedelta(days=1)
    return days

def expand_days_list(days_info: List[Union[str, List[str]]]) -> List[str]:
    days_all = []
    for days in days_info:
        if isinstance(days, str):
            days_all.append(days)
        elif isinstance(days, list):
            days_all += expand_date_range(days)
    return days_all


def get_seg_annot_meta(wf: np.ndarray,
                       feat: np.ndarray) \
        -> dict:
    assert wf.dtype == 'float32' or wf.dtype == 'float64'
    meta = dict(max_amp=get_seg_amp_metric(wf * MAX_VAL_RAW_AUDIO) / MAX_VAL_RAW_AUDIO,
                max_amp_feat=np.max(feat))
    return meta


def get_annots_df_for_days(days_info: List[Union[str, List[str]]],
                           tags: Optional[List[str]] = None) \
        -> pd.DataFrame:
    """Get DataFrame with annot info """
    sql = "SELECT * FROM annots WHERE"
    for i, days in enumerate(days_info):
        if isinstance(days, str):
            sql += " date(datetime_start) = '{}'".format(days)
        elif isinstance(days, list):
            sql += " date(datetime_start) BETWEEN '{}' AND '{}'".format(*days)
        if i != len(days_info) - 1:
            sql += " OR"

    df = execute_pure_sql(sql)

    if tags is not None:
        idxs = np.any(df['tag'].to_numpy()[:, np.newaxis] == np.array(tags), axis=1)
        df = df.loc[idxs, :]

    return df

def extract_features_for_annots(df_annots: pd.DataFrame,
                                use_cache: bool = False,
                                verbose: bool = False) \
        -> FeatureSet:
    """Extract features for all annotations"""
    fs = FeatureSet()

    num_rows = len(df_annots)
    n = 0
    for _, row in df_annots.iterrows():
        if verbose and np.mod(n, int(np.ceil(num_rows / 10))) == 0:
            print_flush('extract features: {}/{}'.format(n, num_rows))
        cache_fname = get_feat_cache_fname(row['datetime_start'], row['datetime_end'], row['tag'])
        if PRINT_DEBUG:
            print_flush([use_cache, cache_fname, os.path.exists(cache_fname)])
        if use_cache and os.path.exists(cache_fname):
            if PRINT_DEBUG:
                t0 = time.time()
            data_ = load_pickle(cache_fname)
            if PRINT_DEBUG:
                print_flush('time0: ' + str(time.time() - t0))
            X_i, meta_i = data_['X'], data_['meta']
        else:
            X_i, meta_i = extract_features_one(row['datetime_start'], dt_end=row['datetime_end'], tag=row['tag'])[1:]
        if use_cache and not os.path.exists(cache_fname):
            save_pickle(cache_fname, dict(X=X_i, meta=meta_i), make_dir=True)
        fs.append(X_i, row['tag'], get_times_from_dts(row['datetime_start']), meta_i)
        n += 1

    return fs

def get_feat_cache_fname(datetime_start,
                         datetime_end = None,
                         tag: Optional[str] = None) \
        -> str:
    """Get feature file name from annot timestamps and tag"""
    def repl(s):
        return s.replace(' ', 's').replace('.', 'p').replace(':', 'c')
    day_dir = str(datetime_start.date())
    cache_fname = repl(str(datetime_start))
    if datetime_end is not None:
        cache_fname += '_' + repl(str(datetime_end))
    if tag is not None:
        cache_fname += '_' + tag
    cache_fpath = os.path.join(FEAT_DIR, day_dir, cache_fname) + '.pickle'
    return cache_fpath

def extract_features_for_segs(days_info: Optional[List[Union[str, List[str]]]] = None,
                              dts: Optional[List[str]] = None,
                              annot_free: bool = False,
                              use_cache: bool = False,
                              max_num_feats: Optional[int] = None,
                              verbose: bool = False) \
        -> FeatureSet:
    """Extract features for all relevant segs"""
    fs = FeatureSet()

    def _extract(df):
        extract_features_for_segs_loop(fs, df, use_cache=use_cache, verbose=verbose)

    if days_info is not None:
        days = expand_days_list(days_info)
        num_days = len(days)
        df = get_seg_df_for_days(days, annot_free=annot_free)
        if df is None:
            return fs
        if max_num_feats is not None:
            num_segs = int(META_FILT_FACTOR_BG * np.ceil(max_num_feats / MAX_NUM_SPEC_FEATS))
        else:
            num_segs = num_days * MAX_BG_SEGS_PER_DAY
        idx = np.random.permutation(len(df))[:num_segs]
        df = df.iloc[idx, :]
        # df = df.sort_values(by=['datetime'])
        _extract(df)

    if dts is not None:
        df = get_seg_df_for_dts(dts)
        _extract(df)

    return fs

def get_seg_df_for_days(days: List[str],
                        annot_free: bool = False) \
        -> Optional[pd.DataFrame]:
    assert isinstance(days, list)
    if len(days) == 0:
        return None

    dfs = []
    for day in days:
        sql = "SELECT * FROM raw WHERE date(datetime) = '" + day + "'"
        df_ = execute_pure_sql(sql)
        if annot_free: # only include segments that don't overlap with an annotation
            seg_idxs = select_annot_free_segs(df_, day)
            df_ = df_.iloc[seg_idxs, :]
        dfs.append(df_)

    df = pd.concat(dfs, axis=0)

    return df

def select_annot_free_segs(df: pd.DataFrame,
                           day: str) \
        -> List[int]:
    sql = "SELECT * FROM annots where date(datetime_start) = '" + day + "'"
    df_annots = execute_pure_sql(sql)

    row_idxs = []
    for i in range(len(df)):#np.random.permutation(len(df)):
        dt_end = df.loc[i, 'datetime']
        dt_start = dt_end - TD_BUFF_DUR
        if ~((dt_start > df_annots['datetime_start']) * (dt_start < df_annots['datetime_end'])).any() and \
                ~((df_annots['datetime_start'] > dt_start) * (df_annots['datetime_start'] < dt_end)).any():
            row_idxs.append(i)
        # if len(row_idxs) == MAX_BG_SEGS_PER_DAY:
        #     break

    return row_idxs

def get_seg_df_for_dts(dts: List[str]) -> pd.DataFrame:
    num_dts = len(dts)
    batch_size = 100
    num_batches = int(np.ceil(num_dts / batch_size))

    dfs = []
    for n in range(num_batches):
        dts_batch = dts[n * batch_size:min(num_dts, (n + 1) * batch_size)]
        sql = "SELECT * FROM raw WHERE"
        for i, dt in enumerate(dts_batch):
            sql += " datetime = '" + dt + "'"
            if i != len(dts_batch) - 1:
                sql += " OR"
        sql += " ORDER BY datetime ASC"
        df_ = execute_pure_sql(sql)
        dfs.append(df_)

    df = pd.concat(dfs, axis=0)

    return df

def extract_features_for_segs_loop(fs: FeatureSet,
                                   df: pd.DataFrame,
                                   use_cache: bool = False,
                                   verbose: bool = False):
    def compute_feats(row):
        X_i, meta_i = extract_features_one(row['datetime'])[1:]
        return X_i, meta_i

    num_rows = len(df)
    n = 0
    for i, row in df.iterrows():
        if verbose and np.mod(n, int(np.ceil(num_rows / 10))) == 0:
            print_flush('extract features: {}/{}'.format(n, num_rows))
        # print_flush(str(n) + ', ' + str(num_rows))
        cache_fname = get_feat_cache_fname(row['datetime'])
        if PRINT_DEBUG:
            print_flush([use_cache, cache_fname, os.path.exists(cache_fname)])
        if use_cache and os.path.exists(cache_fname):
            try:
                if PRINT_DEBUG:
                    t0 = time.time()
                data_ = load_pickle(cache_fname)
                if PRINT_DEBUG:
                    print_flush('time1: ' + str(time.time() - t0))
            except:
                print_flush('Failed to load {}. Re-computing seg feature file.'.format(cache_fname))
                X_i, meta_i = compute_feats(row)
                data_ = dict(X=X_i, meta=meta_i)
                save_pickle(cache_fname, data_, make_dir=True)
            X_i, meta_i = data_['X'], data_['meta']
        else:
            X_i, meta_i = compute_feats(row)
        if use_cache and not os.path.exists(cache_fname):
            save_pickle(cache_fname, dict(X=X_i, meta=meta_i), make_dir=True)
        fs.append(X_i, OTHER_TAG, get_times_from_dts(row['datetime']), meta_i)
        n += 1


def load_macroseg(df_raw: pd.DataFrame) -> np.ndarray:
    fpaths = [row['filepath'] for _, row in df_raw.iterrows()]
    return np.concatenate([read_wav(fpath)[1] for fpath in fpaths])

def extract_features_one(dt_start: datetime,
                         dt_end: Optional[datetime] = None,
                         tag: Optional[str] = None) \
        -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract features from raw waveform segment"""
    # get relevant segs from raw table
    dt_start_s = dt_start.strftime(FMT_DT)
    if dt_end is None:
        sql = "SELECT * FROM raw WHERE datetime = '" + dt_start_s + "'"
    else:
        dt_end_s = (dt_end + TD_BUFF_DUR).strftime(FMT_DT)
        sql = "SELECT * FROM raw WHERE datetime BETWEEN '" + dt_start_s + "' AND '" + dt_end_s + "' ORDER BY datetime ASC"
    df_raw = execute_pure_sql(sql)

    # get macroseg
    macroseg = load_macroseg(df_raw) / MAX_VAL_RAW_AUDIO

    # infer samp start and end within macroseg
    if dt_end is None:
        samp_start = 0
        samp_end = len(macroseg) - 1
    else:
        dt_start_macroseg = df_raw.loc[0, 'datetime'] - TD_BUFF_DUR
        t_start = (dt_start - dt_start_macroseg).total_seconds()
        t_end = (dt_end - dt_start_macroseg).total_seconds()
        samp_start = max(0, int(t_start * SAMPLERATE))
        samp_end = min(int(t_end * SAMPLERATE), len(macroseg) - 1)

    # extract wf
    # wf = macroseg[samp_start:samp_end]

    # vis
    if 0:
        max_amp = np.max(np.abs(macroseg))

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(len(macroseg)), macroseg, c='b')
        for s in [samp_start, samp_end]:
            ax.plot([s] * 2, [-max_amp, max_amp], c='r')
        plt.title(str(dt_start) + '\n' + str(dt_end) + '\n')
        plt.show()

    # featurize it
    wf, feats = featurize(macroseg, samp_start, samp_end, tag=tag)

    # compute additional annot/macroseg info (e.g. for filtering)
    meta = get_seg_annot_meta(wf, feats)

    # print((get_seg_amp_metric(wf_annot), filt_info['max_amp_feat']))

    return wf, feats, meta

def get_features_for_days(days_info: List[Union[str, List[str]]],
                          tags: Optional[List[str]] = None,
                          all_segs: bool = False,
                          annot_free: bool = False,
                          use_cache: bool = False,
                          verbose: bool = False,
                          max_num_feats: Optional[int] = None) \
        -> FeatureSet:
    """
    Feature loader.

    Input args:
        days_info: list of combined day strings ('2022-12-10') and day string ranges (['2022-12-13', '2022-12-15'])
    """
    if not all_segs:
        df_annots = get_annots_df_for_days(days_info, tags=tags) # get dataframe of annots info
        feat_set = extract_features_for_annots(df_annots, use_cache=use_cache, verbose=verbose) # get feature vectors and labels for all annotations found
    else:
        feat_set = extract_features_for_segs(days_info=days_info, annot_free=annot_free, use_cache=use_cache,
                                             max_num_feats=max_num_feats, verbose=verbose)
    return feat_set

def get_features_for_dts(dts: List[str],
                         use_cache: bool = False,
                         verbose: bool = True) \
        -> FeatureSet:
    """Used for real-time detection"""
    return extract_features_for_segs(dts=dts, use_cache=use_cache, verbose=verbose)

