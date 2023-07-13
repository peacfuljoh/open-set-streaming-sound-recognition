"""Recognition models (feature projection/selection + classification)"""

from typing import List, Optional, Tuple, Union

import numpy as np

from sklearn.preprocessing import StandardScaler

from src.ml.classifiers import ClassifierGMM, ClassifierSklearn
from src.ml.transformers import TransformerLDA, TransformerPCA, TransformerStandardScaler, TransformerSklearn, \
    TransformerLDASklearn
from src.ml.transformers_nn import TransformerDNN
from ossr_utils.misc_utils import get_colors, tags_str_to_int, print_flush, get_majority_str
from src.constants_ml import OTHER_TAG





class RecognitionModel():
    def __init__(self, verbose: bool = False):
        self._verbose = verbose

        self._transformer = None
        self._classifier = None

        self._tags: np.ndarray = None  # array of strings
        self._num_tags: int = None


    """Fit"""
    def fit(self,
            feats: np.ndarray,
            tags_list: List[str],
            tods: Optional[np.ndarray] = None,
            feats_bg: Optional[np.ndarray] = None,
            tods_bg_train: Optional[np.ndarray] = None,
            group_idxs: Optional[np.ndarray] = None,
            group_idxs_bg: Optional[np.ndarray] = None):
        raise NotImplementedError

    def _init_model(self,
                    feats: np.ndarray,
                    tags_list: List[str],
                    feats_bg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Validate inputs and save a few pieces of information (orig data dim, tags)"""
        # check inputs
        self._D_orig = feats.shape[1]
        assert isinstance(tags_list, list) and all([isinstance(tag_, str) for tag_ in tags_list])

        # add bg data tags to tags list
        if feats_bg is not None:
            N_bg = feats_bg.shape[0]
            tags_list = tags_list + [OTHER_TAG] * N_bg

        # get list of unique tags
        self._tags = np.unique(tags_list) # scrambles order!
        self._num_tags = len(self._tags)

        # get integer tags and sample counts across all tags
        y = np.where(np.array(tags_list)[:, np.newaxis] == self._tags)[1]
        samp_counts = np.sum(y[:, np.newaxis] == np.arange(self._num_tags), axis=0)

        return y, samp_counts


    """Predict"""
    def predict(self,
                feats: np.ndarray,
                force_det: bool = False,
                tods: Optional[np.ndarray] = None) \
            -> List[str]:
        """Classify feature vectors given current model"""
        # compute recognition metric
        feats_proj = self._transformer.transform(feats)
        metric = self._classifier.get_metric(feats_proj, tods=tods)  # num_samps x num_tags, negative log probs p(t|x)
        if force_det:  # force classification (ignore background model)
            detects = np.ones(metric.shape)
        else:
            detects = metric <= self._classifier.get_det_threshs()

        # set prediction tags
        num_samps = feats.shape[0]
        y_pred = np.ones(num_samps, dtype='int') * self._num_tags
        idxs_det = np.any(detects, axis=1)
        y_pred[idxs_det] = np.argmin(metric[idxs_det, :], axis=1)

        tags_w_ooc = np.concatenate([self._tags, np.array([OTHER_TAG])])
        tags_pred = tags_w_ooc[y_pred]

        return tags_pred

    def predict_multi(self,
                      feats: List[np.ndarray],
                      force_det: bool = False,
                      tods: Optional[List[float]] = None,
                      pred_mask: Optional[np.ndarray] = None) \
            -> List[str]:
        """Majority-vote prediction on batch of feature vectors"""
        N = len(feats)

        if tods is not None:
            assert N == len(tods)

        preds = []
        for i in range(N):
            if pred_mask is not None and not pred_mask[i]:
                preds.append(OTHER_TAG)
                continue
            if tods is not None:
                N_batch = len(feats[i])
                tods_i = np.ones(N_batch) * tods[i] # tod is fixed for all vecs in batch
            else:
                tods_i = None
            pred_i = get_majority_str(self.predict(feats[i], force_det=force_det, tods=tods_i))
            preds.append(pred_i)

        return preds
        # return [get_majority_str(self.predict(f, force_det=force_det, tods=tods)) for f in feats]


    """Print"""
    def _print_train_msg(self, feats, samp_counts):
        print_flush('\n=== SoundRecognitionModel (train) ===')
        print_flush('Total sample count: ' + str(feats.shape[0]))
        print_flush('Per-class (K={}) sample counts: '.format(self._num_tags))
        for k in range(self._num_tags):
            print_flush('  {}: {}'.format(self._tags[k], samp_counts[k]))

    def _print_predict_msg(self, feats):
        print_flush('\n=== SoundRecognitionModel (predict) ===')
        print_flush('Total sample count: ' + str(feats.shape[0]))




class RecognitionModelLDA(RecognitionModel):
    def __init__(self,
                 D_proj_lda: Optional[int] = None,
                 D_proj_pca: Optional[int] = None,
                 verbose: bool = False):
        super().__init__(verbose=verbose)

        self._D_orig = None
        self._D_proj_lda = D_proj_lda
        self._D_proj_pca = D_proj_pca # inferred

        self._A: np.ndarray = None # projector
        self._mu_train: np.ndarray = None # training data mean


    """Fit to data"""
    def fit(self,
            feats: np.ndarray,
            tags_list: List[str],
            tods: Optional[np.ndarray] = None,
            feats_bg: Optional[np.ndarray] = None,
            tods_bg: Optional[np.ndarray] = None,
            group_idxs: Optional[np.ndarray] = None,
            group_idxs_bg: Optional[np.ndarray] = None):
        """Fit recognition model to in-class and out-of-class (background) examples data"""
        ### Setup ###
        # validate input
        assert (feats_bg is None and group_idxs_bg is None) or \
               (feats_bg is not None and group_idxs_bg is not None)
        if feats_bg is not None:
            assert (tods is None and tods_bg is None) or (tods is not None and tods_bg is not None)

        # init fit info
        y = self._init_model(feats, tags_list, feats_bg=feats_bg) # already stacked if feats_bg is not None

        ### Fit model ###
        # projector
        bg_tag_idx, y_tr, num_tags_tr = _prepare_for_transformer_fit(y, feats_bg, self._tags, self._num_tags)
        self._transformer = TransformerLDA(self._D_orig, self._D_proj_pca, self._D_proj_lda, num_tags_tr)
        self._transformer.fit(feats, y_tr)

        # classifier
        feats_cl, group_idxs_cl, tods_cl = \
            _prepare_for_classifier_fit(feats, feats_bg, group_idxs, group_idxs_bg, tods, tods_bg)
        self._classifier = ClassifierGMM(self._num_tags, num_clusters=20, num_iters_cluster=200,
                                         tag_labels=list(self._tags))
        _fit_classifier_w_transform(self, feats_cl, y, bg_y_idx=bg_tag_idx, group_idxs=group_idxs_cl, tods=tods_cl)

    def _init_model(self,
                    feats: np.ndarray,
                    tags_list: List[str],
                    feats_bg: Optional[np.ndarray] = None) -> np.ndarray:
        y, samp_counts = super()._init_model(feats, tags_list, feats_bg=feats_bg)

        if self._D_proj_pca is None:
            self._D_proj_pca = min(self._D_orig, min(10 * self._num_tags, feats.shape[0] / self._num_tags))
        if self._D_proj_lda is None:
            self._D_proj_lda = self._num_tags - 1

        if self._verbose:
            self._print_train_msg(feats, samp_counts)

        # check counts
        lda_samps_cond = feats.shape[0] > self._D_proj_pca
        clust_samps_cond = np.all(samp_counts > self._D_proj_lda)
        assert lda_samps_cond # enough samps for LDA (protects C_x)
        assert clust_samps_cond # enough samps for clusters fit
        # assert self._D_proj_lda < self._num_tags # ensure multiclass LDA is feasible (no empty dimensions)

        return y


    """Messages"""
    def _print_train_msg(self, feats, samp_counts):
        super()._print_train_msg(feats, samp_counts)
        print_flush('Dims: ' + str(feats.shape[1]) + ' -> ' + str(self._D_proj_pca) + ' -> ' + str(self._D_proj_lda))


    """Plot"""
    def plot(self,
             feats_train: np.ndarray,
             feats_bg_train: Optional[np.ndarray] = None,
             feats_test: Optional[np.ndarray] = None):
        assert self._D_proj_lda == 2

        tags_all = list(self._tags)

        import matplotlib.pyplot as plt

        colors = get_colors(20, mult=1.0)[5:5 + (self._num_tags + 1), :]
        colors[-1, :] = np.array([0, 0, 0])  # outlier color

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)

        feats_proj = self._transformer.transform(feats_train)
        pred_feats = tags_str_to_int(self.predict(feats_train), self._num_tags, tags_all=tags_all)
        colors_ = colors[pred_feats, :]

        if feats_bg_train is not None:
            feats_bg_proj = self._transformer.transform(feats_bg_train)
            pred_feats_bg = tags_str_to_int(self.predict(feats_bg_train), self._num_tags, tags_all=tags_all)
            colors_bg_ = colors[pred_feats_bg, :]

        if feats_test is not None:
            feats_test_proj = self._transformer.transform(feats_test)
            pred_feats_test = tags_str_to_int(self.predict(feats_test), self._num_tags, tags_all=tags_all)
            colors_test_ = colors[pred_feats_test, :]

        scat_feat = ax.scatter(feats_proj[:, 0], feats_proj[:, 1], c=colors_, s=10, marker='o')
        if feats_bg_train is not None:
            scat_feat_bg = ax.scatter(feats_bg_proj[:, 0], feats_bg_proj[:, 1], c=colors_bg_, s=1, marker='o')
        if feats_test is not None:
            scat_feat_test = ax.scatter(feats_test_proj[:, 0], feats_test_proj[:, 1], c=colors_test_, s=10, marker='x')
        for k in range(self._num_tags):
            mus_k = self._classifier._cluster_params['mu'][k]
            ax.scatter(mus_k[:, 0], mus_k[:, 1], c='r', marker='+', s=50)
        ax.set_aspect('equal')

        plt.show()






class RecognitionModelSKLearn(RecognitionModel):
    def __init__(self,
                 verbose: bool = False):
        super().__init__(verbose=verbose)

    """Fit to data"""
    def fit(self,
            feats: np.ndarray,
            tags_list: List[str],
            tods: Optional[np.ndarray] = None,
            feats_bg: Optional[np.ndarray] = None,
            tods_bg: Optional[np.ndarray] = None,
            group_idxs: Optional[np.ndarray] = None,
            group_idxs_bg: Optional[np.ndarray] = None):
        """Fit recognition model to in-class and out-of-class (background) examples data"""
        ### Setup ###
        # validate input
        assert (feats_bg is None and group_idxs_bg is None) or \
               (feats_bg is not None and group_idxs_bg is not None)
        if feats_bg is not None:
            assert (tods is None and tods_bg is None) or (tods is not None and tods_bg is not None)

        # init fit info
        y = self._init_model(feats, tags_list, feats_bg=feats_bg)  # already stacked if feats_bg is not None

        ### Fit model ###
        # projector
        bg_tag_idx, y_tr, num_tags_tr = _prepare_for_transformer_fit(y, feats_bg, self._tags, self._num_tags)
        self._transformer = TransformerSklearn(self._D_orig)
        self._transformer.fit(feats, y_tr)

        # classifier
        feats_cl, group_idxs_cl, tods_cl = \
            _prepare_for_classifier_fit(feats, feats_bg, group_idxs, group_idxs_bg, tods, tods_bg)
        # self._classifier = ClassifierSklearn(self._model_type, self._model_params, self._num_tags,
        #                                      tag_labels=list(self._tags))
        self._classifier = ClassifierGMM(self._num_tags, num_clusters=20, num_iters_cluster=200,
                                         tag_labels=list(self._tags))
        _fit_classifier_w_transform(self, feats_cl, y, bg_y_idx=bg_tag_idx, group_idxs=group_idxs_cl, tods=tods_cl)

    def _init_model(self,
                    feats: np.ndarray,
                    tags_list: List[str],
                    feats_bg: Optional[np.ndarray] = None) -> np.ndarray:
        y, samp_counts = super()._init_model(feats, tags_list, feats_bg=feats_bg)

        if self._verbose:
            self._print_train_msg(feats, samp_counts)

        return y


    """Messages"""
    def _print_train_msg(self, feats, samp_counts):
        super()._print_train_msg(feats, samp_counts)









class RecognitionModelDNN(RecognitionModel):
    def __init__(self,
                 verbose: bool = False):
        super().__init__(verbose=verbose)

    """Fit to data"""
    def fit(self,
            feats: np.ndarray,
            tags_list: List[str],
            tods: Optional[np.ndarray] = None,
            feats_bg: Optional[np.ndarray] = None,
            tods_bg: Optional[np.ndarray] = None,
            group_idxs: Optional[np.ndarray] = None,
            group_idxs_bg: Optional[np.ndarray] = None):
        """Fit recognition model to in-class and out-of-class (background) examples data"""
        ### Setup ###
        # validate input
        assert (feats_bg is None and group_idxs_bg is None) or \
               (feats_bg is not None and group_idxs_bg is not None)
        if feats_bg is not None:
            assert (tods is None and tods_bg is None) or (tods is not None and tods_bg is not None)

        # init fit info
        y = self._init_model(feats, tags_list, feats_bg=feats_bg)  # already stacked if feats_bg is not None

        ### Fit model ###
        # projector
        bg_tag_idx, y_tr, num_tags_tr = _prepare_for_transformer_fit(y, feats_bg, self._tags, self._num_tags)
        self._transformer = TransformerDNN(num_tags_tr)
        self._transformer.fit(feats, y_tr)

        # classifier
        feats_cl, group_idxs_cl, tods_cl = \
            _prepare_for_classifier_fit(feats, feats_bg, group_idxs, group_idxs_bg, tods, tods_bg)
        self._classifier = ClassifierGMM(self._num_tags, num_clusters=20, num_iters_cluster=200,
                                         tag_labels=list(self._tags))
        _fit_classifier_w_transform(self, feats_cl, y, bg_y_idx=bg_tag_idx, group_idxs=group_idxs_cl, tods=tods_cl)

    def _init_model(self,
                    feats: np.ndarray,
                    tags_list: List[str],
                    feats_bg: Optional[np.ndarray] = None) -> np.ndarray:
        y, samp_counts = super()._init_model(feats, tags_list, feats_bg=feats_bg)

        if self._verbose:
            self._print_train_msg(feats, samp_counts)

        return y


    """Messages"""
    def _print_train_msg(self, feats, samp_counts):
        super()._print_train_msg(feats, samp_counts)








"""Helper functions"""
def _fit_classifier_w_transform(model: RecognitionModel,
                                feats: np.ndarray,
                                y: np.ndarray,
                                bg_y_idx: Optional[int] = None,
                                group_idxs: Optional[np.ndarray] = None,
                                tods: Optional[np.ndarray] = None):
    feats_proj = model._transformer.transform(feats)
    model._classifier.fit(feats_proj, y, bg_y_idx=bg_y_idx, group_idxs=group_idxs, tods=tods)

def _prepare_for_transformer_fit(y: np.ndarray,
                                 feats_bg: Union[np.ndarray, None],
                                 tags: np.ndarray,
                                 num_tags: int) \
        -> Tuple[Optional[int], np.ndarray, int]:
    if feats_bg is not None: # remove background info for estimating projector
        bg_tag_idx = np.where(tags == OTHER_TAG)[0][0]
        y_tr = y[y != bg_tag_idx]  # only non-bg tag indices
        y_tr[y_tr > bg_tag_idx] -= 1  # OTHER_TAG might not be last in tag list
        num_tags_tr = num_tags - 1
    else:
        bg_tag_idx = None
        y_tr = y
        num_tags_tr = num_tags
    return bg_tag_idx, y_tr, num_tags_tr

def _prepare_for_classifier_fit(feats: np.ndarray,
                                feats_bg: Union[np.ndarray, None],
                                group_idxs,
                                group_idxs_bg,
                                tods: Union[np.ndarray, None],
                                tods_bg) \
        -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if feats_bg is not None:
        feats_cl = np.vstack((feats, feats_bg))
        num_groups = np.max(group_idxs)
        group_idxs_cl = np.concatenate((group_idxs, group_idxs_bg + num_groups))
        if tods is not None:
            tods_cl = np.concatenate((tods, tods_bg))
        else:
            tods_cl = None
    else:
        feats_cl = feats
        group_idxs_cl = None
        tods_cl = tods
    return feats_cl, group_idxs_cl, tods_cl