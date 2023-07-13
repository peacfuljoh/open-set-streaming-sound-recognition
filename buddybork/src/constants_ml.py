'''
Constants related to ML
'''


import os

import numpy as np

from src.constants_stream import SAMPLERATE
from ossr_utils.io_utils import load_json

# dirs
PATHS_FPATH = os.environ['BB_PATHS']

PATHS = load_json(PATHS_FPATH)
FEAT_DIR = PATHS['FEAT_DIR']
MODEL_DIR = PATHS['MODEL_DIR']
MODEL_JSON_FPATH = os.path.join(MODEL_DIR, 'models.json')

# misc
OTHER_TAG = 'other'
MAX_BG_SEGS_PER_DAY = 500

APPLY_META_FILT = True
if APPLY_META_FILT:
    # MIN_SEG_MAX_AMP_DB = -40
    MIN_SEG_MAX_AMP_DB = -35
    MIN_META_MAX_AMP = 10 ** (MIN_SEG_MAX_AMP_DB / 20) # float value
    # META_FILT_FACTOR_BG = 5 # approximate bg data reduction factor due to filtering
    META_FILT_FACTOR_BG = 15 # approximate bg data reduction factor due to filtering
else:
    MIN_META_MAX_AMP = 0
    META_FILT_FACTOR_BG = 1

# featurization
WIN_SIZE = 1024
OVERLAP_FACTOR = 2
HOP_SIZE = int(WIN_SIZE / OVERLAP_FACTOR)
NUM_FEATS_FFT = WIN_SIZE // 2 + 1
NUM_FEATS_TOT = NUM_FEATS_FFT + 1 # add one for waveshape

# MAX_DUR_SPEC_FEATS = 0.50 # seconds
MAX_DUR_SPEC_FEATS = 0.25 # seconds
MAX_NUM_SPEC_FEATS = int(MAX_DUR_SPEC_FEATS / (HOP_SIZE / SAMPLERATE)) # max number of feature vectors

# time model
TM_MIN_VAL = 0 # min value
TM_MAX_VAL = 24 # max value
grid_points_per_hour = 3
TM_GRID_LEN = TM_MAX_VAL * grid_points_per_hour + 1
TM_GRID = np.linspace(TM_MIN_VAL, TM_MAX_VAL, TM_GRID_LEN)
TM_KERN_VAR = 0.5 ** 2 # time model kernel variance
# TM_BASELINE_PROB = 0.1 # baseline time model value
TM_MAX_KERN_VAL = 10 # max pre-norm KDE value
TM_MIN_KERN_VAL = 1 # min pre-norm KDE value
TM_MIN_NUM_SAMPS = 20 # min samps to fit KDE
TM_PROB_DEFAULT = TM_MIN_KERN_VAL / TM_MAX_KERN_VAL # default probability when too few samples to fit KDE

# recognition model
D_PROJ_PCA = 200