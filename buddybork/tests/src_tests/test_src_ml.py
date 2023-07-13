
import copy

import numpy as np

from src.ml.classifiers import ClassifierGMM
from src.ml.features import FeatureSet
from src.ml.featurization import featurize, get_waveshape_feats
from src.ml.ml_models import RecognitionModelLDA
from src.ml.transformers import TransformerPCA, TransformerLDA

from src.constants_ml import NUM_FEATS_TOT, D_PROJ_PCA

from tests.utils_for_tests import make_toy_feats, make_wavs




def test_classifier_gmm():
    X, y, K, group_idxs, tods, _, _, _, _ = make_toy_feats()
    N = X.shape[0]

    bg_y_idx = 0

    args_opts = [
        dict(bg_y_idx=None, group_idxs=None, tods=None),
        dict(bg_y_idx=bg_y_idx, group_idxs=group_idxs, tods=None),
        dict(bg_y_idx=bg_y_idx, group_idxs=group_idxs, tods=tods),
        dict(bg_y_idx=None, group_idxs=None, tods=tods)
    ]

    for opt in args_opts:
        classifier = ClassifierGMM(K,
                                   cluster_cov_mode='scaled_identity',
                                   num_clusters=3,
                                   num_iters_cluster=5)

        classifier.fit(X,
                       y,
                       bg_y_idx=opt['bg_y_idx'],
                       group_idxs=opt['group_idxs'],
                       tods=opt['tods'],
                       set_threshs=True)

        # assert cluster params
        for k in range(K):
            for key in ['mu', 'var', 'w']:
                assert isinstance(classifier._cluster_params[key][k], np.ndarray)

        # assert time model
        if opt['tods'] is not None:
            assert isinstance(classifier._time_model, np.ndarray)

        # assert decision thresholds
        assert isinstance(classifier._det_threshs, np.ndarray)

        # check metric
        for k in [None, 0]:
            metric = classifier.get_metric(X, k, tods=opt['tods'])
            assert isinstance(metric, np.ndarray)
            if k is None:
                assert metric.shape == (N, K)
            else:
                assert metric.shape == (N, )

def test_feature_set():
    num_tags = 2
    n_per_tag = 5
    num_tot = num_tags * n_per_tag
    num_feats = 3

    feats = [np.arange(n_per_tag * num_feats).reshape(n_per_tag, num_feats) + num_tags * i for i in range(num_tags)]
    tags_ = 'abcdefghij'
    tags = [tags_[i] for i in range(num_tags)]
    tods = np.random.rand(num_tags) * 24
    metas = [dict(max_amp=0) for _ in range(num_tags)]

    fs = FeatureSet()
    for i in range(num_tags):
        fs.append(feats[i], tags[i], tods[i], metas[i])

    # test get_Xy_individual()
    Xy_ind = fs.get_Xy_individual()

    for i in range(num_tags):
        assert np.allclose(Xy_ind[0][i], feats[i])
        assert Xy_ind[1][i] == tags[i]
        assert Xy_ind[2][i] == tods[i]

    # test get_Xy_full()
    Xy = fs.get_Xy_full()

    assert np.allclose(Xy[0], np.vstack(feats))
    tags_exp = tags[0:1] * n_per_tag + tags[1:2] * n_per_tag
    assert all([Xy[1][i] == tags_exp[i] for i in range(num_tot)])
    group_idxs_exp = np.kron(np.arange(num_tags), np.ones(n_per_tag))
    assert all([Xy[2][i] == group_idxs_exp[i] for i in range(num_tot)])
    assert np.allclose(Xy[3], np.kron(tods, np.ones(n_per_tag)))

    # test merge()
    fs2 = copy.deepcopy(fs)
    fs2.merge(copy.deepcopy(fs))

    # test filter_by_meta()
    for apply in [False, True]:
        keep_ = fs.filter_by_meta(apply=apply)
        assert len(keep_) == num_tags
        assert all([out == False for out in keep_])

    Xy_ind = fs.get_Xy_individual()
    assert all([out == [] for out in Xy_ind])
    Xy = fs.get_Xy_full()
    assert all([out is None for out in Xy])

def test_featurization():
    samp_start = 100
    samp_end = 5000

    wfs, tags = make_wavs()

    wf_annot, feats = featurize(wfs[0], samp_start, samp_end, tag=tags[0])

    assert len(wf_annot) == (samp_end - samp_start)
    assert feats.shape[1] == NUM_FEATS_TOT

    spec_amps, spec_amps_smooth, peak_idxs_valid = get_waveshape_feats(feats)

    assert isinstance(spec_amps, np.ndarray)
    assert isinstance(spec_amps_smooth, np.ndarray)
    assert isinstance(peak_idxs_valid, list)

def test_recognition_model_lda():
    X, y, K, group_idxs, tods, tags, X_bg, group_idxs_bg, tods_bg = make_toy_feats()

    model = RecognitionModelLDA(D_proj_pca=D_PROJ_PCA)
    model.fit(X, tags)
    model.predict(X)
    model.predict_multi([X] * 2)

    model = RecognitionModelLDA(D_proj_pca=D_PROJ_PCA)
    model.fit(X, tags, tods=tods)
    model.predict(X, tods=tods)
    model.predict_multi([X] * 2, tods=list(tods[:2]))

    model = RecognitionModelLDA(D_proj_pca=D_PROJ_PCA)
    model.fit(X, tags, feats_bg=X_bg, group_idxs=group_idxs, group_idxs_bg=group_idxs_bg)
    model.predict(X)
    model.predict_multi([X] * 2)

    model = RecognitionModelLDA(D_proj_pca=D_PROJ_PCA)
    model.fit(X, tags, feats_bg=X_bg, group_idxs=group_idxs, group_idxs_bg=group_idxs_bg, tods=tods, tods_bg=tods_bg)
    model.predict(X, tods=tods)
    model.predict_multi([X] * 2, tods=list(tods[:2]))

def test_transformer():
    X, y, K, group_idxs, tods, tags, X_bg, _, _ = make_toy_feats()

    # PCA
    tr = TransformerPCA(NUM_FEATS_TOT, D_PROJ_PCA)
    tr.fit(X)
    assert tr._A.shape == (NUM_FEATS_TOT, D_PROJ_PCA)
    assert len(tr._mu) == NUM_FEATS_TOT

    tr = TransformerPCA(NUM_FEATS_TOT, D_PROJ_PCA)
    tr.fit(X, X_bg=X_bg)
    assert tr._A.shape == (NUM_FEATS_TOT, D_PROJ_PCA)
    assert len(tr._mu) == NUM_FEATS_TOT

    X_proj = tr.transform(X)
    assert X_proj.shape == (X.shape[0], D_PROJ_PCA)

    # LDA
    tr = TransformerLDA(NUM_FEATS_TOT, D_PROJ_PCA, K - 1, K)
    tr.fit(X, y)
    assert tr._A.shape == (NUM_FEATS_TOT, K - 1)
    assert len(tr._mu) == NUM_FEATS_TOT

    tr = TransformerLDA(NUM_FEATS_TOT, D_PROJ_PCA, K - 1, K)
    tr.fit(X, y, X_bg=X_bg)
    assert tr._A.shape == (NUM_FEATS_TOT, K - 1)
    assert len(tr._mu) == NUM_FEATS_TOT

    X_proj = tr.transform(X)
    assert X_proj.shape == (X.shape[0], K - 1)





if __name__ == '__main__':
    test_classifier_gmm()
    test_feature_set()
    test_featurization()
    test_recognition_model_lda()
    test_transformer()
