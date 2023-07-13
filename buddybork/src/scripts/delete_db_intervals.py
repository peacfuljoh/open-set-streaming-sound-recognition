"""Script for redaction"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils.db_utils import execute_pure_sql, execute_pure_sql_no_return
from src.scripts.db_intervals import INTERVALS


NUM_SEGS_BUFF = 5
N_SHIFT_FILT = (NUM_SEGS_BUFF - 1) // 2

PROMPT_BEFORE_DELETE = False
CHECK_FOR_LONG_INTERVALS = False


def detect_yes_input(in_: str) -> bool:
    return in_.strip(' |\n|\t') in ['y', 'yes']


if not PROMPT_BEFORE_DELETE or not CHECK_FOR_LONG_INTERVALS:
    print('Some safety macros are turned off. Are you sure?')
    print('PROMPT_BEFORE_DELETE: ' + str(PROMPT_BEFORE_DELETE))
    print('CHECK_FOR_LONG_INTERVALS: ' + str(CHECK_FOR_LONG_INTERVALS))
    in_ = input('Proceed (y/yes/n/no)? ')
    if not detect_yes_input(in_):
        exit(0)

str_fmt = '%Y-%m-%d %H:%M:%S'

for day, intervals in INTERVALS.items():
    for interval in intervals:
        # get dt info
        dt_start = day + ' ' + interval[0] + ':00'
        dt_end = day + ' ' + interval[1] + ':00'

        dt_start_d = datetime.strptime(dt_start, str_fmt)
        dt_end_d = datetime.strptime(dt_end, str_fmt)
        dt_diff = dt_end_d - dt_start_d

        if CHECK_FOR_LONG_INTERVALS and dt_diff.total_seconds() > 5 * 60 * 60:
            raise Exception('Redaction interval must be shorter than 5 hours.')
        if dt_end_d <= dt_start_d:
            raise Exception('Start time must come before end time.')

        # get all segs in interval
        sql = "SELECT * FROM raw WHERE datetime BETWEEN '{}' AND '{}' ORDER BY datetime ASC".format(dt_start, dt_end)
        df_segs = execute_pure_sql(sql)
        num_segs = len(df_segs)
        if num_segs == 0:
            continue

        # get all annots in interval
        sql = "SELECT * FROM annots WHERE datetime_start BETWEEN '{}' AND '{}'".format(dt_start, dt_end)
        df_annots = execute_pure_sql(sql)

        # identify segs that overlap with annots
        idxs_drm = pd.Series(False, index=df_segs.index)
        # print(df_segs[(df_segs['datetime'] >= '2022-12-30 13:08:25') * (df_segs['datetime'] <= '2022-12-30 13:08:35')])
        for _, row in df_annots.iterrows():
            dt_start_ = row['datetime_start']
            dt_end_ = row['datetime_end']

            # annot interval includes seg dt
            idxs_0 = (df_segs['datetime'] >= dt_start_) * (df_segs['datetime'] <= dt_end_)

            # annot interval contained between seg dts
            idxs_1 = (df_segs['datetime'].iloc[:-1].reset_index(drop=True) <= dt_start_) * \
                     (df_segs['datetime'].iloc[1:].reset_index(drop=True) >= dt_end_)

            # add annot-overlapping seg idxs to mask
            # print([np.sum(idxs_0), np.sum(idxs_1)])
            idxs_drm |= idxs_0
            idxs_drm[1:] |= idxs_1

        # expand annot masks
        idxs_drm = np.convolve(idxs_drm.to_numpy(), np.ones(NUM_SEGS_BUFF))
        idxs_rm = (idxs_drm == 0)[N_SHIFT_FILT:N_SHIFT_FILT + num_segs]

        # show and warn
        print('Redaction:')
        print('  interval: [{}, {}]'.format(dt_start, dt_end))
        print('  duration: ' + str(dt_diff))
        print('  num segs: ' + str(len(df_segs)))
        print('  num annots: ' + str(len(df_annots)))

        # print(df_segs)
        # print(df_annots)

        if 0:
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.scatter(df_segs['datetime'], idxs_rm)
            plt.show()

        # redact
        if PROMPT_BEFORE_DELETE:
            in_ = input('Continue (y/yes/n/no)?')
        else:
            in_ = 'y'
        if detect_yes_input(in_):
            for i in np.where(idxs_rm)[0]:
                row = df_segs.iloc[i]
                # print(row['filepath'])
                # print(row['datetime'])

                # remove files
                if os.path.exists(row['filepath']):
                    os.remove(row['filepath'])

                # remove rows from raw table
                sql = "DELETE FROM raw WHERE datetime = '{}'".format(row['datetime'])
                execute_pure_sql_no_return(sql)
        else:
            print('Skipping this one.\n')








