'''
Constants related to data capture and storage
'''

import time
import os

from ossr_utils.io_utils import load_json


# session start info
TS_INIT = int(1e3 * time.time()) # UTC ms

# paths
PATHS_FPATH = os.environ['BB_PATHS']
PATHS = load_json(PATHS_FPATH)
REPO_ROOT = os.environ['BB_ROOT']
DATA_DIR = PATHS['DATA_DIR']
CONSTANTS_PATH = os.path.join(REPO_ROOT, 'constants.json')
CAPTURE_STATE_JSON_PATH = PATHS['CAPTURE_STATE_JSON_PATH']

# audio capture constants
const_json = load_json(CONSTANTS_PATH)
DEVICE_NAME = const_json['DEVICE_NAME']
NUM_CHANNELS = const_json['NUM_CHANNELS']
SAMPLERATE = const_json['SAMPLERATE']
BLOCKSIZE = const_json['BLOCKSIZE']
NUM_BLOCKS_IN_BUFF = const_json['NUM_BLOCKS_IN_BUFF']
BUFF_LEN = const_json['BUFF_LEN']
BUFF_DUR = const_json['BUFF_DUR']

# get local config
def getLocalConfig(configPath, defaultObj):
    if os.path.exists(configPath):
        return load_json(configPath)
    print('Could not find config file: ' + configPath)
    return defaultObj

DB_CONFIG = getLocalConfig(PATHS['DB_CONFIG_PATH'], {})
NET_CONFIG = getLocalConfig(PATHS['NET_CONFIG_PATH'], {'HUB_IP': '0', 'SERVER_PORT': 0})

# other
MAX_VAL_RAW_AUDIO = 2 ** 15 - 1

MAX_AMP_THRESH_DB = -48 # 130 sig val
MAX_AMP_THRESH = int(10 ** (MAX_AMP_THRESH_DB / 20) * MAX_VAL_RAW_AUDIO) # int16
LEN_DATA_SESS_DIR = 13 # UTC us string
