'''
Class for conveniently managing features and groups of features
'''

from typing import List, Tuple, Union

import numpy as np

from src.constants_ml import MIN_META_MAX_AMP


class FeatureSet():
    def __init__(self):
        self._feats: List[np.ndarray] = [] # feature matrices
        self._tags: List[str] = [] # tags for each matrix
        self._tods: List[float] = [] # time-of-day
        self._metas: List[dict] = [] # meta info
        self._f: List[int] = [] # tag idxs (grouping index when feats is vstack'd)
        self._n: int = 0 # number of feat matrices

    def append(self,
               feats: np.ndarray,
               tag: str,
               tod: float,
               meta: dict):
        num_samps = feats.shape[0]
        self._feats.append(feats)
        self._tags.append(tag)
        self._tods.append(tod)
        self._metas.append(meta)
        self._f += [self._n] * num_samps
        self._n += 1

    def get_Xy_full(self) \
            -> Union[Tuple[np.ndarray, List[str], np.ndarray, np.ndarray],
                     Tuple[None, None, None, None]]:
        if self._n > 0:
            X = np.vstack(self._feats) # all features in one array
            y = [self._tags[i] for i in self._f] # all tag strings in one list
            t = [self._tods[i] for i in self._f] # all time-of-day values in one line
            return X, y, np.array(self._f), np.array(t)
        else:
            return None, None, None, None

    def get_Xy_individual(self):
        return self._feats, self._tags, self._tods

    def merge(self, fs):
        N = fs._n
        self._feats += fs._feats
        self._tags += fs._tags
        self._tods += fs._tods
        self._metas += fs._metas
        self._f += [f_ + N for f_ in fs._f]
        self._n += N

    def filter_by_meta(self, apply: bool = True) -> np.ndarray:
        """Determine which entries to keep based on metadata filtering criteria. Optionally apply in-place."""
        keep = np.zeros(self._n, dtype='bool')
        for i in range(self._n):
            amp_cond = self._metas[i]['max_amp'] >= MIN_META_MAX_AMP
            keep[i] = amp_cond
        if apply:
            fs_new = FeatureSet()
            for i in range(self._n):
                if keep[i]:
                    fs_new.append(self._feats[i], self._tags[i], self._tods[i], self._metas[i])
            for key in ['feats', 'tags', 'tods', 'metas', 'f', 'n']:
                self.__dict__['_' + key] = fs_new.__dict__['_' + key]
        return keep

    def summary(self) -> dict:
        return dict(
            num_FVs=sum([feat.shape[0] for feat in self._feats]),
            num_mats=self._n,
            num_tags=len(np.unique(self._tags))
        )
