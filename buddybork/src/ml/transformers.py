'''
Feature transformation (e.g. via projection)
'''

from typing import Optional, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning



class Transformer():
    def __init__(self,
                 D_orig: int):
        self._D_orig = D_orig
        self._mu = None
        self._A = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project feature vectors"""
        return (X - self._mu) @ self._A


class TransformerLDA(Transformer):
    def __init__(self,
                 D_orig: int,
                 D_proj_pca: int,
                 D_proj_lda: int,
                 K: int):
        super().__init__(D_orig)
        self._D_proj_pca = D_proj_pca
        self._D_proj_lda = D_proj_lda
        self._K = K

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_bg: Optional[np.ndarray] = None):
        # get original mean and subtract it off
        self._mu = np.mean(X, axis=0)
        X = X - self._mu
        if X_bg is not None:
            X_bg = X_bg - self._mu

        # PCA
        transformer_pca = TransformerPCA(self._D_orig, self._D_proj_pca)
        transformer_pca.fit(X, X_bg=X_bg)
        A_pca = transformer_pca._A
        X = X @ A_pca
        if X_bg is not None:
            X_bg = X_bg @ A_pca

        # LDA
        A_lda = self._calc_LDA_projector(X, y, X_bg=X_bg)

        # set overall projector
        self._A = A_pca @ A_lda

    def _calc_LDA_projector(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            X_bg: Optional[np.ndarray] = None) \
            -> np.ndarray:
        C_x, C_m = self._calc_covs(X, y)
        u, s, vh = np.linalg.svd(np.linalg.solve(C_x, C_m))
        A = u[:, :self._D_proj_lda]
        return A

    def _calc_covs(self,
                   X: np.ndarray,
                   y: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        # cluster means
        mus = np.zeros((self._K, self._D_proj_pca), dtype='float')
        for k in range(self._K):
            mus[k, :] = np.mean(X[y == k, :], axis=0)

        # within-class covariance
        C_x = np.zeros((self._D_proj_pca, self._D_proj_pca), dtype='float')
        for k in range(self._K):
            xm = X[y == k, :] - mus[k, :]
            C_x += xm.T @ xm
        C_x /= (X.shape[0] - self._K)

        # between-class covariance
        mm = mus - np.mean(X, axis=0)
        samp_counts = np.sum(np.arange(self._K)[:, np.newaxis] == y, axis=1)
        C_m = (mm.T * samp_counts) @ mm / (self._K - 1)

        return C_x, C_m



class TransformerPCA(Transformer):
    def __init__(self,
                 D_orig: int,
                 D_proj_pca: int):
        super().__init__(D_orig)

        self._D_proj_pca = D_proj_pca

    def fit(self,
            X: np.ndarray,
            X_bg: Optional[np.ndarray] = None):
        # get original mean and subtract it off
        self._mu = np.mean(X, axis=0)
        X = X - self._mu
        if X_bg is not None:
            X_bg = X_bg - self._mu

        # set projector
        self._A = self._calc_PCA_projector(X, X_bg=X_bg)

    def _calc_PCA_projector(self,
                            X: np.ndarray,
                            X_bg: Optional[np.ndarray] = None) \
            -> np.ndarray:
        if self._D_proj_pca < self._D_orig:
            u, s, vh = np.linalg.svd(X, full_matrices=False)
            A_pca = vh[:self._D_proj_pca, :].T * (1 / s[:self._D_proj_pca])
        else:
            A_pca = np.eye(self._D_proj_pca)
        return A_pca



class TransformerLDASklearn(Transformer):
    def __init__(self,
                 D_orig: int,
                 D_proj_pca: int,
                 D_proj_lda: int):
        super().__init__(D_orig)
        self._D_proj_pca = D_proj_pca
        self._D_proj_lda = D_proj_lda

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_bg: Optional[np.ndarray] = None):
        # PCA
        self._transformer_pca = PCA(self._D_proj_pca)
        self._transformer_pca.fit(X)
        X = self._transformer_pca.transform(X)

        # LDA
        self._transformer_lda = LinearDiscriminantAnalysis(n_components=self._D_proj_lda)
        self._transformer_lda.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = self._transformer_pca.transform(X)
        X_lda = self._transformer_lda.transform(X_pca)
        return X_lda





class TransformerStandardScaler(Transformer):
    def __init__(self, D_orig=None):
        super().__init__(D_orig)

    def fit(self, X: np.ndarray):
        self._mu = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._std



class TransformerSklearn(Transformer):
    def __init__(self, D_orig):
        super().__init__(D_orig)

        self._offset = None
        self._coef = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        self._preprocessor = StandardScaler()
        self._preprocessor.fit(X)
        X = self._preprocessor.transform(X)

        # The convergence warning from sklearn is related to early exit in the 'trainer/model/fit/:data' route
        # (i.e. premature return in pymodel.py 'fit' method with 'undefined' instead of dict(modelId=model_id)).
        # Turning off the convergence warning fixes this...
        simplefilter("ignore", category=ConvergenceWarning)

        self._classifier = LogisticRegression(penalty='l2', tol=1e-4, C=1e-2, solver='lbfgs', max_iter=500,
                                              multi_class='multinomial', verbose=0)
        self._classifier.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocessor.transform(X)
        return X @ self._classifier.coef_.T + self._classifier.intercept_
