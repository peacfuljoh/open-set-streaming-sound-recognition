
import json
import os
from typing import Tuple

from src.constants_ml import MODEL_JSON_FPATH, MODEL_DIR
from ossr_utils.io_utils import load_json
from ossr_utils.misc_utils import print_flush


def finish(result):
    '''Final print of result info before exiting Python sub-process'''
    print_flush('pyout:' + json.dumps(result))


def get_model_info_from_id(model_id: str) -> Tuple[str, str, str]:
    '''Get various model info from model ID string'''
    models = load_json(MODEL_JSON_FPATH)
    if model_id not in models:
        raise Exception('Model ID {} not found in library'.format(model_id))
    model = models[model_id]
    return model['filename'], os.path.join(MODEL_DIR, model['filename']), model_id
