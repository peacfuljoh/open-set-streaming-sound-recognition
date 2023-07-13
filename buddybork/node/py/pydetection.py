"""Detection script to be called from Express app"""

from typing import List
import sys
import json

PYTHON_REPO_ROOT = sys.argv[1]
sys.path.append(PYTHON_REPO_ROOT)

from src.ml.features import FeatureSet
from src.utils.feats_io_utils import get_features_for_dts
from ossr_utils.io_utils import load_pickle
from ossr_utils.misc_utils import print_flush

from node.py.pyfunc_utils import finish, get_model_info_from_id


MODE_OPTS = ['detect']
USE_FEAT_CACHE = True



def detect(dts: List[str],
           model_id: str) \
        -> dict:
    # get pre-trained model info
    _, model_cache_fpath, _ = get_model_info_from_id(model_id)
    print_flush('Loading model info, id = ' + str(model_id))
    model = load_pickle(model_cache_fpath)['model']

    # get test and background data
    feat_set: FeatureSet = get_features_for_dts(dts, use_cache=USE_FEAT_CACHE, verbose=True)

    # get features
    feats_test, _, tods_test = feat_set.get_Xy_individual()

    # evaluate model on test data
    pred_mask = feat_set.filter_by_meta(apply=False)
    tags_pred: List[str] = model.predict_multi(feats_test, tods=tods_test, pred_mask=pred_mask)

    return dict(tags=tags_pred)



def main():
    mode = sys.argv[2]

    assert mode in MODE_OPTS

    params = json.loads(sys.argv[3])
    if mode == 'detect':
        finish(detect(params['dts'], params['model_id']))
    else:
        raise Exception('Specify valid mode option from ' + str(MODE_OPTS))


if __name__ == "__main__":
    main()