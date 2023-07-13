

import os
import time
from datetime import datetime

from ossr_utils.io_utils import load_json


PATHS_FPATH = os.environ['BB_PATHS']
PATHS = load_json(PATHS_FPATH)
ASSETS_DIR = PATHS['ASSETS_DIR']


clean_flag = True
while 1:
    print("Initiated cleanup process")
    if 2 <= datetime.now().hour <= 3: # every night after 2 AM
        if clean_flag:
            fpaths = [os.path.join(ASSETS_DIR, fname) for fname in sorted(os.listdir(ASSETS_DIR))
                      if 'macroseg' in fname and '.wav' in fname]
            print('Deleting temporary asset files (' + str(datetime.now().date()) + ')')
            for fpath in fpaths:
                print('   ' + fpath)
                os.remove(fpath)
            clean_flag = False
    else:
        clean_flag = True
    time.sleep(5 * 60) # check every 5 minutes
