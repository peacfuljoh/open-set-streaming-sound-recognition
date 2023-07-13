"""Fit model script to be called from Express app"""

from typing import Dict, List, Tuple, Optional
import sys
import json
import os
import time

import numpy as np

PYTHON_REPO_ROOT = sys.argv[1]
sys.path.append(PYTHON_REPO_ROOT)

from src.ml.ml_models import RecognitionModelLDA, RecognitionModelSKLearn
from src.ml.features import FeatureSet
from src.constants_ml import OTHER_TAG, MODEL_JSON_FPATH, MODEL_DIR, D_PROJ_PCA
from src.utils.feats_io_utils import get_features_for_days
from ossr_utils.io_utils import load_pickle, save_pickle, load_json, save_json
from ossr_utils.model_utils import get_conf_mat
from ossr_utils.misc_utils import arr_to_lists, print_flush

from node.py.pyfunc_utils import finish, get_model_info_from_id


MODEL_TYPE = 'sklearn'
# MODEL_TYPE = 'lda'
MODE_OPTS = ['fit', 'predict', 'delete']
USE_FEAT_CACHE = True



def parse_params(params: dict) -> Tuple[Dict[str, List[str]], Optional[List[str]], Optional[str], str]:
    # extract days
    days = dict(R=[], E=[], Br=[], Be=[])
    for day_, group_ in params['days'].items():
        days[group_].append(day_)

    # extract tags
    try:
        tags = params['tags']
        if len(tags) == 0:
            tags = None
    except:
        tags = None

    # model id
    try:
        model_id = params['model_id']
    except:
        model_id = None

    # model name
    model_name = params['model_name']

    return days, tags, model_id, model_name

def get_model_info_from_days_and_tags(days_train: List[str],
                                      days_bg: List[str],
                                      tags: List[str]) \
        -> Tuple[str, str, str]: # fname, fpath, model_id
    # load up existing model info
    models = load_json(MODEL_JSON_FPATH)

    # check if any existing models match the specified params
    for id, info in models.items():
        days_train_cond = set(info['days_train']) == set(days_train)
        days_bg_cond = set(info['days_bg']) == set(days_bg)
        tags_cond = (info['tags'] is None and tags is None) or \
                    ((info['tags'] is not None and tags is not None) and (set(info['tags']) == set(tags)))
        if days_train_cond and days_bg_cond and tags_cond:
            return info['filename'], os.path.join(MODEL_DIR, info['filename']), id

    # didn't find a matching model
    model_id = str(int(1e3 * time.time()))
    new_model_fname = model_id + '.pickle'
    return new_model_fname, os.path.join(MODEL_DIR, new_model_fname), model_id



def fit(days_train: List[str],
        days_bg: List[str],
        tags: List[str],
        model_name: str) \
        -> dict:
    # get model info
    model_cache_fname, model_cache_fpath, model_id = get_model_info_from_days_and_tags(days_train, days_bg, tags)

    # fit model
    if not os.path.exists(model_cache_fpath):
        # fit new model
        print_flush('Fitting model')
        print_flush('days_train: ' + str(days_train))
        print_flush('days_bg: ' + str(days_bg))
        print_flush('tags: ' + str(tags))
        print_flush('model name: ' + model_name)

        feat_set_train: FeatureSet = get_features_for_days(days_train, use_cache=USE_FEAT_CACHE, verbose=True, tags=tags)
        feat_set_train.filter_by_meta()
        feats_train, tags_train, group_idxs, tods_train = feat_set_train.get_Xy_full()

        if len(days_bg) > 0:
            N_avg = int(np.mean(np.sum(np.array(tags_train)[:, np.newaxis] == np.unique(tags_train), axis=0)))
            feat_set_bg_train: FeatureSet = get_features_for_days(days_bg, all_segs=True, use_cache=USE_FEAT_CACHE,
                                                                  annot_free=True, verbose=True, max_num_feats=N_avg)
            feat_set_bg_train.filter_by_meta()
            feats_bg, _, group_idxs_bg, tods_bg = feat_set_bg_train.get_Xy_full()
        else:
            feats_bg = None
            group_idxs_bg = None
            tods_bg = None

        print_flush('Train annot count: ' + str(feat_set_train._n))

        if MODEL_TYPE == 'lda':
            model = RecognitionModelLDA(D_proj_pca=D_PROJ_PCA)
        elif MODEL_TYPE == 'sklearn':
            model = RecognitionModelSKLearn()
        else:
            raise RuntimeError
        model.fit(feats_train, tags_train, tods=tods_train,
                  feats_bg=feats_bg, tods_bg=tods_bg,
                  group_idxs=group_idxs, group_idxs_bg=group_idxs_bg)

        # save model and info
        print_flush('Saving model info, id = ' + str(model_id))
        save_pickle(model_cache_fpath, dict(model=model))
        models = load_json(MODEL_JSON_FPATH)
        models[model_id] = dict(days_train=days_train, days_bg=days_bg, tags=tags,
                                filename=model_cache_fname, model_name=model_name)
        save_json(MODEL_JSON_FPATH, models)

        print_flush('Exiting training for model_id ' + model_id)

    return dict(modelId=model_id)


def predict(days_test: List[str],
            days_bg: List[str],
            model_id: str) \
        -> dict:
    # get pre-trained model info
    _, model_cache_fpath, _ = get_model_info_from_id(model_id)
    print_flush('Loading model info, id = ' + str(model_id))
    model = load_pickle(model_cache_fpath)['model']

    # get test and background data
    has_test = len(days_test) > 0
    has_bg = len(days_bg) > 0
    if has_test:
        print_flush('has_test: ' + str(days_test))
        feat_set_test: FeatureSet = get_features_for_days(days_test, use_cache=USE_FEAT_CACHE, verbose=True)
    if has_bg:
        print_flush('has_bg: ' + str(days_bg))
        feat_set_bg: FeatureSet = get_features_for_days(days_bg, use_cache=USE_FEAT_CACHE, verbose=True,
                                                        all_segs=True, annot_free=True)
    if has_test and not has_bg:
        feat_set = feat_set_test
    elif not has_test and has_bg:
        feat_set = feat_set_bg
    elif has_test and has_bg:
        feat_set = feat_set_test
        feat_set.merge(feat_set_bg)
    else:
        raise Exception('Specify test and/or bg days for prediction.')

    # get features
    feats_test, tags_test_true, tods_test = feat_set.get_Xy_individual()  # List[np.ndarray], List[str]

    # get train and test tag lists
    tags_train = list(model._tags)
    if OTHER_TAG not in tags_train:
        tags_train.append(OTHER_TAG)
    tags_train = sorted(tags_train)
    tags_test = sorted(list(np.unique(tags_test_true)))

    # evaluate model on test data
    pred_mask = feat_set.filter_by_meta(apply=False)
    tags_test_pred = model.predict_multi(feats_test, tods=tods_test, pred_mask=pred_mask)
    conf_mat = get_conf_mat(tags_train, tags_test, tags_test_true, tags_test_pred)

    print_flush('Exiting prediction with model_id ' + model_id)

    return dict(confMat=arr_to_lists(conf_mat), tagsTrain=tags_train, tagsTest=tags_test)


def delete(model_id: str) -> dict:
    models = load_json(MODEL_JSON_FPATH)
    if model_id in models:
        print_flush('Deleting model, id = ' + str(model_id))
        model_cache_fpath = os.path.join(MODEL_DIR, models[model_id]['filename'])
        os.remove(model_cache_fpath)
        del models[model_id]
        save_json(MODEL_JSON_FPATH, models)
    else:
        print_flush('Model not found in library, id = ' + str(model_id))

    print_flush('Exiting deletion for model_id ' + model_id)

    return dict()


def main():
    try:
        mode = sys.argv[2]

        assert mode in MODE_OPTS
        params = json.loads(sys.argv[3])
        if mode == 'delete':
            finish(delete(params['model_id']))
        else:
            days, tags, model_id, model_name = parse_params(params)
            if mode == 'fit':
                finish(fit(days['R'], days['Br'], tags, model_name))
            elif mode == 'predict':
                finish(predict(days['E'], days['Be'], model_id))
            else:
                raise Exception('Specify valid mode option from ' + str(MODE_OPTS))
    except Exception as inst:
        print_flush("Error: {}".format(inst))


if __name__ == "__main__":
    main()

