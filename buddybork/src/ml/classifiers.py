'''
Classification models and time model fit
'''

from typing import Optional, Tuple, Union, List

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from ossr_utils.misc_utils import get_dists, logsumexp, gauss_pdf
from src.constants_ml import TM_GRID_LEN, TM_MAX_VAL, TM_MIN_NUM_SAMPS, TM_PROB_DEFAULT, TM_GRID, TM_KERN_VAR, \
    TM_MAX_KERN_VAL, TM_MIN_KERN_VAL



SHOW_PREC_RECALL_CURVES_IN_SET_DET_THRESH = False # debug switch (visualization)


class Classifier():
    """
    Parent class for all classifiers.

    A new child class must implement, the following methods: __init__(), fit(), _compute_metric_one()
    """
    def __init__(self,
                 K: int,
                 tag_labels: Optional[List[str]] = None):
        self._K = K
        self._tag_labels = tag_labels

        self._det_threshs = None

        self._min_recall = 0.6 # true positive rate = correct detections / num positives
        # self._min_precision = 0.9 # 1 - false positive rate = correct detections / all detections

        self._time_model: np.ndarray = None

    def fit(self,
            X: np.ndarray,  # data matrix (vectors in rows)
            y: np.ndarray,  # class indices
            bg_y_idx: Optional[int] = None,
            group_idxs: Optional[np.ndarray] = None,
            set_threshs: bool = True):
        raise NotImplementedError

    def _set_det_threshs(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         bg_y_idx: Optional[int] = None,
                         group_idxs: Optional[np.ndarray] = None,
                         tods: Optional[np.ndarray] = None):
        assert (bg_y_idx is None and group_idxs is None) or (bg_y_idx is not None and group_idxs is not None)

        self._det_threshs = np.zeros(self._K, dtype='float')

        if bg_y_idx is None:
            for k in range(self._K):
                self._set_det_threshs_no_bg(X, y, k, tods=tods)
        else:
            prec_all = []
            recall_all = []
            for k in range(self._K):
                prec, recall = self._set_det_threshs_w_bg(X, y, group_idxs, bg_y_idx, k, tods=tods)
                prec_all.append(prec)
                recall_all.append(recall)
            if SHOW_PREC_RECALL_CURVES_IN_SET_DET_THRESH:
                self._plot_prec_recall(prec_all, recall_all, group_idxs, bg_y_idx, y)

        if bg_y_idx is not None:
            self._det_threshs[bg_y_idx] = -np.Inf # never explicitly detect with background model

    def _set_det_threshs_no_bg(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               k: int,
                               tods: Optional[np.ndarray] = None):
        """Doesn't use grouping info"""
        idxs_k = y == k
        if tods is not None:
            tods_k = tods[idxs_k]
        else:
            tods_k = None
        metric = self.get_metric(X[idxs_k, :], k, tods=tods_k)
        N_k = len(metric)
        idx_cutoff = int(np.round(self._min_recall * (N_k - 1)))
        self._det_threshs[k] = np.sort(metric)[idx_cutoff]

    def _set_det_threshs_w_bg(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              group_idxs: np.ndarray,
                              bg_y_idx: int,
                              k: int,
                              tods: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        # calculate precision and recall info from detection counts in increasing metric order
        metric_threshs, prec, recall = self._get_prec_recall_curves(X, y, group_idxs, bg_y_idx, k, tods=tods)

        # set det thresh
        idx_min_recall = np.where(recall >= self._min_recall)[0][0]
        idx_max_prec = idx_min_recall + np.argmax(prec[idx_min_recall:])
        if idx_max_prec > idx_min_recall: # max prec thresh lies beyond min recall thresh
            idx_thresh = idx_max_prec
        else: # find max recall beyond init recall thresh without sacrificing prec
            prec_at_min_recall = prec[idx_min_recall]
            idx_thresh = idx_min_recall + np.max(np.where(prec[idx_min_recall:] >= 0.95 * prec_at_min_recall)[0])
        self._det_threshs[k] = metric_threshs[idx_thresh]

        return prec, recall

    def _get_prec_recall_curves(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                group_idxs: np.ndarray,
                                bg_y_idx: int,
                                k: int,
                                tods: Optional[np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate precision and recall info from detection counts in increasing metric order"""
        # get in-class and background data subsets
        idxs_k = y == k
        idxs_bg = y == bg_y_idx
        idxs_all = idxs_k | idxs_bg
        X_sub = X[idxs_all, :]
        y_sub = y[idxs_all]
        g_sub = group_idxs[idxs_all]
        if tods is not None:
            tods_sub = tods[idxs_all]
        else:
            tods_sub = None

        # calculate precision and recall
        metric = self.get_metric(X_sub, k, tods=tods_sub)
        ord = np.argsort(metric)

        num_groups = np.max(g_sub) + 1
        num_groups_sub = len(np.unique(g_sub))

        dets = np.zeros(num_groups_sub, dtype='float')
        detected = np.zeros(num_groups, dtype='bool') # TODO: remove unused entries
        metric_threshs = np.zeros(num_groups_sub, dtype='float')

        i = 0  # index into dets
        for j in ord:  # ordering of feats for increasing metric value
            group_idx_j = g_sub[j]
            det_thresh_j = metric[j]
            dets_j = metric[g_sub == group_idx_j] <= det_thresh_j
            if np.mean(dets_j) > 0.5:  # group-level detection
                if not detected[group_idx_j]:  # new detection
                    detected[group_idx_j] = True
                    metric_threshs[i] = metric[j]
                    dets[i] = y_sub[j] == k
                    i += 1
        # dets = y_sub[ord] == k # alternative to above that doesn't use grouping

        counts = np.cumsum(np.vstack((1 - dets, dets)).T, axis=0)
        prec = counts[:, 1] / np.sum(counts, axis=1)
        recall = counts[:, 1] / counts[-1, 1]

        return metric_threshs, prec, recall

    def _plot_prec_recall(self,
                          prec_all: List[np.ndarray],
                          recall_all: List[np.ndarray],
                          group_idxs: np.ndarray,
                          bg_y_idx: int,
                          y: np.ndarray):
        import matplotlib.pyplot as plt

        plots_per_row = 4
        num_rows = int(np.ceil(self._K / plots_per_row))

        fig = plt.figure(figsize=(8, 8))
        axes = fig.subplots(num_rows, plots_per_row, sharex=True, sharey=True, squeeze=False)
        for k in range(self._K):
            if k == bg_y_idx:
                continue
            i = int(k / plots_per_row)
            j = k - i * plots_per_row
            axes[i, j].plot(recall_all[k], prec_all[k], c='k')
            axes[i, j].set_xlim([-0.02, 1.02])
            axes[i, j].set_ylim([-0.02, 1.02])
            if i == num_rows - 1:
                axes[i, j].set_xlabel('Recall')
            if j == 0:
                axes[i, j].set_ylabel('Precision')
            num_groups_k = len(set(list(group_idxs[y == k])))
            axes[i, j].set_title(self._tag_labels[k] + ' (' + str(num_groups_k) + ')')
        plt.show()

    def get_metric(self,
                   X: np.ndarray,
                   k: Optional[int] = None,
                   tods: Optional[np.ndarray] = None) \
            -> np.ndarray:
        if k is None:
            N = X.shape[0]
            metric = np.zeros((N, self._K), dtype='float')
            for j in range(self._K):
                metric[:, j] = self._compute_metric_one(X, j, tods=tods)
        else:
            metric = self._compute_metric_one(X, k, tods=tods)
        return metric

    def _compute_metric_one(self,
                            X: np.ndarray,
                            k: int,
                            tods: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """
        Compute distance/dissimilarity metric from a data batch to a specified tag's model.
        This is typically an un-normalized negative log probability for one tag's model, i.e. - log p(t=k|x).
        """
        raise NotImplementedError

    def _compute_metric_one_tods(self,
                                 tods: np.ndarray,
                                 k: int) \
            -> np.ndarray:
        """Compute log probability for time-of-day model for a given tag"""
        grid_idxs = ((tods / TM_MAX_VAL) * TM_GRID_LEN).astype('int')
        p_t = np.log(self._time_model[grid_idxs, k])
        return p_t

    def get_det_threshs(self) -> np.ndarray:
        return self._det_threshs



class ClassifierGMM(Classifier):
    def __init__(self,
                 K: int,
                 cluster_cov_mode: str = 'scaled_identity',
                 num_clusters: int = 30,
                 num_iters_cluster: int = 200,
                 tag_labels: Optional[List[str]] = None):
        super().__init__(K, tag_labels=tag_labels)

        self._num_clusters = num_clusters
        self._cluster_cov_mode = cluster_cov_mode # 'scaled_identity', 'shared_scaled_identity'

        self._prior_weights = np.zeros(K, dtype='float')
        self._cluster_params = dict(mu=[None] * K, var=[None] * K, w=[None] * K)

        self._num_iters_cluster = num_iters_cluster

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            bg_y_idx: Optional[int] = None,
            group_idxs: Optional[np.ndarray] = None,
            tods: Optional[np.ndarray] = None,
            set_threshs: bool = True):
        """
        Fit clustering model to each available data class.

        Inputs:
            X: (num_samps x num_feats) data matrix
            y: (num_samps) class label indices corresponding to rows of data matrix
            bg_y_idx: index of background class
            group_idxs: (num_samps) indices indicating what feature vectors are grouped into a single "sample"
            tods: (num_samps) time-of-day feature values
            set_threshs: option to set decision thresholds after clustering
        """
        N = X.shape[0]
        N_y = len(y)
        assert N == N_y

        # determine prior weights based on tag counts
        self._prior_weights = compute_prior_weights(y, self._K)

        # fit mixture models
        for k in range(self._K):
            if 0:
                w_k, mus_k, var_k = self._fit_clusters(X[y == k, :], self._num_clusters)
            else:
                gmm_k = GaussianMixture(n_components=self._num_clusters, covariance_type='spherical', tol=1e-3,
                                        reg_covar=1e-6, max_iter=self._num_iters_cluster, n_init=1,
                                        init_params='kmeans')
                gmm_k.fit(X[y == k, :])
                w_k, mus_k, var_k = gmm_k.weights_, gmm_k.means_, gmm_k.covariances_
            self._cluster_params['mu'][k] = mus_k
            self._cluster_params['var'][k] = var_k
            self._cluster_params['w'][k] = w_k

        # fit time model
        if tods is not None:
            self._time_model = fit_time_model_all(y, tods, self._K)

        # set detection thresholds
        if set_threshs:
            self._set_det_threshs(X, y, bg_y_idx=bg_y_idx, group_idxs=group_idxs, tods=tods)

    def _fit_clusters(self,
                      x: np.ndarray,
                      K: int,
                      use_mixing_weights: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
        """Fit cluster model to data"""
        N, D = x.shape

        if N < K:
            raise Exception('Number of samples cannot be less than the number of clusters.')

        # init params
        w = np.ones(K, dtype='float') / K

        mus = np.zeros((K, D), dtype='float')
        mus[0, :] = x[np.random.randint(N), :]
        for k in range(1, K):
            idx_k = np.argmax(np.min(get_dists(x, mus[:k, :]), axis=1))
            mus[k, :] = x[idx_k, :]

        if self._cluster_cov_mode == 'shared_scaled_identity':
            vars = np.median(get_dists(mus, mus) ** 2) # scalar
        elif self._cluster_cov_mode == 'scaled_identity':
            vars = np.median(get_dists(mus, mus) ** 2, axis=0) # 1d array

        # iterate
        for i in range(self._num_iters_cluster):
            p, _, _ = self._e_step(x, w, mus, vars)
            w, mus, vars = self._m_step(x, p, use_mixing_weights)
            w, vars = self._param_protection(w, vars, use_mixing_weights)

        return w, mus, vars

    def _e_step(self,
               x: np.ndarray,
               w: Union[float, np.ndarray],
               mus: np.ndarray,
               vars: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate p(z|x) and log p(x)"""
        D = x.shape[1]
        if self._cluster_cov_mode in ['shared_scaled_identity', 'scaled_identity']:
            p_xz = np.log(w) - D / 2 * np.log(vars) - get_dists(x, mus) ** 2 / (2 * vars) # NxK
        p_x = logsumexp(p_xz, 1)
        p_zx = np.exp(p_xz - p_x) # sum to 1 in the rows
        return p_zx, p_x[:, 0], p_xz

    def _m_step(self,
                x: np.ndarray,
                p: np.ndarray,
                use_mixing_weights: bool):
        D = x.shape[1]
        ps = np.sum(p, axis=0)
        pss = np.sum(ps)
        mus = (p.T @ x) / ps[:, np.newaxis]
        if self._cluster_cov_mode == 'shared_scaled_identity':
            vars = np.sum(p * get_dists(x, mus) ** 2) / (D * pss)
        elif self._cluster_cov_mode == 'scaled_identity':
            vars = np.sum(p * get_dists(x, mus) ** 2, axis=0) / (D * ps)
        if use_mixing_weights:
            w = ps / pss
        return w, mus, vars

    def _param_protection(self,
                          w: np.ndarray,
                          vars: np.ndarray,
                          use_mixing_weights: bool):
        if self._cluster_cov_mode == 'scaled_identity':
            max_var = np.max(vars)
            min_var = 1e-2 * max_var
            vars[vars < min_var] = min_var
        if use_mixing_weights:
            w = np.maximum(w, 1e-3)
            w /= np.sum(w)
        return w, vars

    def _compute_metric_one(self,
                            X: np.ndarray,
                            k: int,
                            tods: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """Compute un-normalized negative log probability for one tag's model, - log p(t=k|x)"""
        params = [self._cluster_params[s][k] for s in ['w', 'mu', 'var']]
        p = np.log(self._prior_weights[k]) + self._e_step(X, *params)[1] # p(t) p(x | t)
        if tods is not None:
            p += self._compute_metric_one_tods(tods, k)
        return -p


    """Tests"""
    def test_fit_class_models(self):
        x = np.random.rand(1000, 2) * np.array([10, 2])
        K = 5
        mus, var = self._fit_clusters(x, K)

        import matplotlib.pyplot as plt
        from ossr_utils.misc_utils import get_colors

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')

        colors = get_colors(K, mult=0.8)

        p = - get_dists(x, mus) ** 2 / (2 * var)
        ps = logsumexp(p, 1)
        p = np.exp(p - ps)
        cc = p @ colors
        ax.scatter(x[:, 0], x[:, 1], c=cc, s=5)
        ax.scatter(mus[:, 0], mus[:, 1], c=colors, marker='+', s=50)

        plt.show()

# class ClassifierGMM(Classifier):
#     def __init__(self,
#                  K: int,
#                  cluster_cov_mode: str = 'scaled_identity',
#                  num_clusters: int = 30,
#                  num_iters_cluster: int = 200,
#                  tag_labels: Optional[List[str]] = None):
#         super().__init__(K, tag_labels=tag_labels)
#
#         self._num_clusters = num_clusters
#         self._cluster_cov_mode = cluster_cov_mode # 'scaled_identity', 'shared_scaled_identity'
#
#         self._prior_weights = np.zeros(K, dtype='float')
#         self._cluster_params = dict(mu=[None] * K, var=[None] * K, w=[None] * K)
#
#         self._num_iters_cluster = num_iters_cluster
#
#     def fit(self,
#             X: np.ndarray,
#             y: np.ndarray,
#             bg_y_idx: Optional[int] = None,
#             group_idxs: Optional[np.ndarray] = None,
#             tods: Optional[np.ndarray] = None,
#             set_threshs: bool = True):
#         """
#         Fit clustering model to each available data class.
#
#         Inputs:
#             X: (num_samps x num_feats) data matrix
#             y: (num_samps) class label indices corresponding to rows of data matrix
#             bg_y_idx: index of background class
#             group_idxs: (num_samps) indices indicating what feature vectors are grouped into a single "sample"
#             tods: (num_samps) time-of-day feature values
#             set_threshs: option to set decision thresholds after clustering
#         """
#         N = X.shape[0]
#         N_y = len(y)
#         assert N == N_y
#
#         # determine prior weights based on tag counts
#         self._prior_weights = compute_prior_weights(y, self._K)
#
#         # fit mixture models
#         for k in range(self._K):
#             gmm_k = GaussianMixture(n_components=self._num_clusters, covariance_type='spherical', tol=1e-3,
#                                     reg_covar=1e-6, max_iter=self._num_iters_cluster, n_init=1, init_params='kmeans',
#                                     )
#             gmm_k.fit(X[y == k, :])
#             w_k, mus_k, var_k = gmm_k.weights_, gmm_k.means_, gmm_k.covariances_
#             # w_k, mus_k, var_k = self._fit_clusters(X[y == k, :], self._num_clusters)
#             self._cluster_params['mu'][k] = mus_k
#             self._cluster_params['var'][k] = var_k
#             self._cluster_params['w'][k] = w_k
#
#         # fit time model
#         if tods is not None:
#             self._time_model = fit_time_model_all(y, tods, self._K)
#
#         # set detection thresholds
#         if set_threshs:
#             self._set_det_threshs(X, y, bg_y_idx=bg_y_idx, group_idxs=group_idxs, tods=tods)
#
#     def _compute_metric_one(self,
#                             X: np.ndarray,
#                             k: int,
#                             tods: Optional[np.ndarray] = None) \
#             -> np.ndarray:
#         """Compute un-normalized negative log probability for one tag's model, - log p(t=k|x)"""
#         params = [self._cluster_params[s][k] for s in ['w', 'mu', 'var']]
#         p = np.log(self._prior_weights[k]) + self._e_step(X, *params)[1] # p(t) p(x | t)
#         if tods is not None:
#             p += self._compute_metric_one_tods(tods, k)
#         return -p
#
#
#     """Tests"""
#     def test_fit_class_models(self):
#         x = np.random.rand(1000, 2) * np.array([10, 2])
#         K = 5
#         mus, var = self._fit_clusters(x, K)
#
#         import matplotlib.pyplot as plt
#         from ossr_utils.misc_utils import get_colors
#
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(1, 1, 1)
#         ax.set_aspect('equal')
#
#         colors = get_colors(K, mult=0.8)
#
#         p = - get_dists(x, mus) ** 2 / (2 * var)
#         ps = logsumexp(p, 1)
#         p = np.exp(p - ps)
#         cc = p @ colors
#         ax.scatter(x[:, 0], x[:, 1], c=cc, s=5)
#         ax.scatter(mus[:, 0], mus[:, 1], c=colors, marker='+', s=50)
#
#         plt.show()



class ClassifierSklearn(Classifier):
    def __init__(self,
                 model_type: str,
                 model_params: dict,
                 K: int,
                 tag_labels: Optional[List[str]] = None):
        super().__init__(K, tag_labels=tag_labels)

        self._model_type = model_type
        self._model_params = model_params

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            bg_y_idx: Optional[int] = None,
            group_idxs: Optional[np.ndarray] = None,
            tods: Optional[np.ndarray] = None,
            set_threshs: bool = True):
        """Fit classifier"""
        N = X.shape[0]
        N_y = len(y)
        assert N == N_y

        # determine prior weights based on tag counts
        self._prior_weights = compute_prior_weights(y, self._K)

        # fit clustering model
        self._classifier = LogisticRegression(penalty='l2', tol=1e-5, C=1, solver='lbfgs', max_iter=500,
                                              multi_class='multinomial', verbose=1)
        self._classifier.fit(X, y)

        # fit time model
        if tods is not None:
            self._time_model = fit_time_model_all(y, tods, self._K)

        # set detection thresholds
        if set_threshs:
            self._set_det_threshs(X, y, bg_y_idx=bg_y_idx, group_idxs=group_idxs, tods=tods)

    def _compute_metric_one(self,
                            X: np.ndarray,
                            k: int,
                            tods: Optional[np.ndarray] = None) \
            -> np.ndarray:
        p = X @ self._classifier.coef_[k, :].T + self._classifier.intercept_[k] # p(x|t)
        p += np.log(self._prior_weights[k]) + p # p(t) p(x|t)
        if tods is not None:
            p += self._compute_metric_one_tods(tods, k)
        return -p









def fit_time_model_all(tods, y, K) -> np.ndarray:
    time_model = np.vstack([fit_time_model(tods[y == k]) for k in range(K)]).T # TM_GRID_LEN x K:
    return time_model

def fit_time_model(x: np.ndarray) -> np.ndarray:
    # fit time-of-day (robust) KDE model to samples
    N = len(x)
    if N < TM_MIN_NUM_SAMPS:
        t = TM_PROB_DEFAULT * np.ones(len(TM_GRID))
    else:
        t = np.zeros(len(TM_GRID), dtype='float')
        for x_ in x:
            t += gauss_pdf(x_, TM_KERN_VAR, TM_GRID, include_norm=False, wrap=[0, 24])
        t = np.minimum(t, TM_MAX_KERN_VAL)
        t = np.maximum(t, TM_MIN_KERN_VAL)
        t /= np.max(t) # normalize
    return t

def compute_prior_weights(y: np.ndarray,
                          K: int) \
        -> np.ndarray:
    """Compute normalized counts"""
    N = len(y)
    return np.sum(y[:, np.newaxis] == np.arange(K), axis=0) / N