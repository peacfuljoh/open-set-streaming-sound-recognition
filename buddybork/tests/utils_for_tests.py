
from typing import Tuple, List

import numpy as np

from src.constants_ml import NUM_FEATS_TOT
from src.constants_stream import SAMPLERATE



np.random.seed(0)

tags_ = 'abcdefghijklmnopqrstuvwxyz'



def make_toy_feats_one(N, K, N_per_k, num_groups, num_samps_per_group, tag_idx_0=0):
    X = np.random.randn(N, NUM_FEATS_TOT)
    y = np.kron(np.arange(K) + tag_idx_0, np.ones(N_per_k)).astype('int')
    tags = [tags_[i] for i in y]

    group_idxs = np.arange(K * num_groups)[:, np.newaxis] * np.ones(num_samps_per_group) + tag_idx_0 * num_groups
    group_idxs = group_idxs.ravel().astype('int')

    return X, y, tags, group_idxs

def make_toy_feats() -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, List[str],
                              np.ndarray, np.ndarray, np.ndarray]:
    K = 10
    num_samps_per_group = 20
    num_groups = 5
    N_per_k = num_samps_per_group * num_groups
    N = N_per_k * K

    X, y, tags, group_idxs = make_toy_feats_one(N, K, N_per_k, num_groups, num_samps_per_group)
    X_bg, _, _, group_idxs_bg = make_toy_feats_one(N_per_k, 1, N_per_k, num_groups, num_samps_per_group, tag_idx_0=K)

    tods = np.random.rand(N) * 24
    tods_bg = np.random.rand(N_per_k) * 24

    return X, y, K, group_idxs, tods, tags, X_bg, group_idxs_bg, tods_bg

def make_wavs() -> Tuple[List[np.ndarray], List[str]]:
    K = 10
    wf_dur = 3.0
    num_samps = int(wf_dur * SAMPLERATE)

    wfs = [np.arange(num_samps) for _ in range(K)]
    tags = [tags_[i] for i in range(K)]

    return wfs, tags


