"""
Script to back up raw data and sql tables for buddybork system.
All other files are cache (features). Annots, tags, etc. are entirely in the tables.
"""

import os
import shutil

import numpy as np


MAX_FILES_PER_DIR = 10000 # drive complains otherwise


UPDATE_BACKUP_RAW_DATA = True
UPDATE_BACKUP_SQL_TABLES = True
UPDATE_BACKUP_CONFIG_FILES = True


dir_nuc = '/home/nuc/buddybork_data'
dir_drive = '/media/nuc/Bork drive'

dbname = 'buddybork'

dir_config_nuc = '/home/nuc/buddybork_config'
dir_config_drive = os.path.join(dir_drive, 'buddybork_config')


# update backup of all data files
if UPDATE_BACKUP_RAW_DATA:
    print('\n=== Updating raw data files ===')

    for dirname in sorted(os.listdir(dir_nuc)):
        # get paths of session dirs
        dpath_nuc = os.path.join(dir_nuc, dirname)
        dpath_drive = os.path.join(dir_drive, 'buddybork_data', dirname)

        # get source file names
        fnames = [fname for fname in sorted(os.listdir(dpath_nuc)) if '.wav' in fname]

        # copy over new files
        print('Updating {} data files for {}'.format(len(fnames), dirname))
        if len(fnames) == 0:
            dpath_drive_i = dpath_drive + '-0'
            if not os.path.exists(dpath_drive_i):
                os.makedirs(dpath_drive_i)
        else:
            for i, fname in enumerate(fnames):
                if np.mod(i, int(0.1 * len(fnames))) == 0:
                    print('  {}/{}'.format(i + 1, len(fnames) + 1))

                # determine destination folder name
                nf = int(np.floor(i / MAX_FILES_PER_DIR))
                dpath_drive_i = dpath_drive + '-' + str(nf)
                if not os.path.exists(dpath_drive_i):
                    os.makedirs(dpath_drive_i)

                # source and destination paths
                fpath_nuc = os.path.join(dpath_nuc, fname)
                fpath_drive = os.path.join(dpath_drive_i, fname)

                # do the copy if the dest file doesn't exist
                if not os.path.exists(fpath_drive):
                    shutil.copy(fpath_nuc, fpath_drive)


# backup postgresql tables
if UPDATE_BACKUP_SQL_TABLES:
    print('\n=== Updating SQL tables ===')

    sql_fname = dbname + '.sql'
    os.system('pg_dump ' + dbname + ' > ' + sql_fname)
    os.system('mv ' + sql_fname + ' "' + dir_drive + '"')

# backup config files
if UPDATE_BACKUP_CONFIG_FILES:
    print('\n=== Updating config files ===')

    if ('home/nuc' in dir_config_drive) or ('media/nuc' not in dir_config_drive):
        raise Exception
    else:
        if os.path.exists(dir_config_drive):
            shutil.rmtree(dir_config_drive)
        shutil.copytree(dir_config_nuc, dir_config_drive)



# CLI tool to manage parallel data directories
# 'rsync -a --update -P /home/nuc/buddybork_data "/media/nuc/Bork drive" --log-file=/home/nuc/Desktop/rsync_log.txt'
