"""Entry point for offline dev of ML model"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from src.ml.ml_models import RecognitionModelLDA, RecognitionModelSKLearn, RecognitionModelDNN
from src.ml.features import FeatureSet
from src.constants_ml import OTHER_TAG, D_PROJ_PCA
from src.utils.feats_io_utils import get_features_for_days
from ossr_utils.model_utils import get_conf_mat



np.random.seed(1)



def main():
    # settings
    use_cache = True

    days_train = [['2022-12-01', '2023-02-13']] # inclusive
    days_bg_train = ['2023-02-14', '2023-02-15']
    # days_bg_train = []
    days_test = [['2023-02-16', '2023-02-20']]
    days_bg_test = ['2023-02-21', '2023-02-22']
    # days_bg_test = []
    tags = [
        'bork',
        'bwrph',
        'crate',
        'dabadab',
        'door_close',
        'keys',
        'kibble',
        'sneeze',
        'snuck',
        'tags_jingle',
    ]

    # model_type = 'lda'
    model_type = 'sklearn'
    # model_type = 'dnn'

    assert model_type in ['lda', 'sklearn', 'dnn']

    # get features
    feat_set_train: FeatureSet = get_features_for_days(days_train, tags=tags, use_cache=use_cache, verbose=True)
    feat_set_train.filter_by_meta()
    feat_set_test: FeatureSet = get_features_for_days(days_test, tags=tags, use_cache=use_cache, verbose=True)

    feat_set_bg_test: FeatureSet = get_features_for_days(days_bg_test, all_segs=True, use_cache=use_cache,
                                                         annot_free=True, verbose=True)

    feat_set_test.merge(feat_set_bg_test)

    feats_train, tags_train, group_idxs_train, tods_train = feat_set_train.get_Xy_full()
    feats_test, tags_test, tods_test = feat_set_test.get_Xy_individual()

    N_avg = int(np.mean(np.sum(np.array(tags_train)[:, np.newaxis] == np.unique(tags_train), axis=0)))
    feat_set_bg_train: FeatureSet = get_features_for_days(days_bg_train, all_segs=True, use_cache=use_cache,
                                                          annot_free=True, verbose=True,
                                                          max_num_feats=N_avg)
    feat_set_bg_train.filter_by_meta()
    feats_bg_train, _, group_idxs_bg_train, tods_bg_train = feat_set_bg_train.get_Xy_full()

    # tods_train, tods_test, tods_bg_train, tods_bg_test = None, None, None, None

    # datasets info
    tags_train_unique = sorted(list(np.unique(tags_train)))
    tags_test_unique = sorted(list(np.unique(tags_test)))

    num_tags_train = len(tags_train_unique)
    num_tags_test = len(tags_test_unique)

    print('Train tags: ' + str(tags_train_unique))
    print('Test tags: ' + str(tags_test_unique))

    print('Train annot count: ' + str(feat_set_train._n))
    print('Train vec count: ' + str(feats_train.shape[0]))
    print('Test annot count: ' + str(feat_set_test._n))

    # train
    if model_type == 'lda':
        model = RecognitionModelLDA(D_proj_pca=D_PROJ_PCA)
    if model_type == 'sklearn':
        model = RecognitionModelSKLearn()
    if model_type == 'dnn':
        model = RecognitionModelDNN()
    model.fit(feats_train, tags_train, tods=tods_train,
              feats_bg=feats_bg_train, tods_bg=tods_bg_train,
              group_idxs=group_idxs_train, group_idxs_bg=group_idxs_bg_train)

    # test
    tags_train_plot = sorted(tags_train_unique + [OTHER_TAG])
    pred_mask = feat_set_test.filter_by_meta(apply=False)
    tags_test_pred = model.predict_multi(feats_test, tods=tods_test, pred_mask=pred_mask)
    conf_mat = get_conf_mat(tags_train_plot, tags_test_unique, tags_test, tags_test_pred)

    # vis
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(conf_mat, cmap=plt.cm.get_cmap('jet'), origin='upper')
    ax.set_xticks(np.arange(num_tags_test) + 0.4)
    ax.set_xticklabels([tag + ' (' + str(np.sum(np.array(tags_test) == tag)) + ')' for tag in tags_test_unique])
    ax.set_yticks(np.arange(num_tags_train + 1))
    ax.set_yticklabels(tags_train_plot)
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=False)
    plt.xticks(rotation=45)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            txt_s = str(conf_mat[i, j])
            left_shift = 0.08 * len(txt_s)
            txt = ax.text(j - left_shift, i + 0.1, txt_s, c='w', fontsize=10, weight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])

    plt.show()





if __name__ == "__main__":
    main()