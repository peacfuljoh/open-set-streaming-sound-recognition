"""Data capture via Flask app"""

import os

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request

from ossr_utils.io_utils import load_json
from ossr_utils.audio_utils import write_wav
from ossr_utils.misc_utils import convert_utc_to_dt
from src.utils.db_utils import insert_or_update_db
from src.constants_stream import SAMPLERATE, DEVICE_NAME, NUM_CHANNELS, BLOCKSIZE, NUM_BLOCKS_IN_BUFF, BUFF_LEN, \
    TS_INIT, DATA_DIR, MAX_AMP_THRESH, BUFF_DUR, CAPTURE_STATE_JSON_PATH, NET_CONFIG


TS_INIT_FLASK = TS_INIT

app = Flask(__name__)


@app.route('/')
def home():
    return '<h3>Hello. The app is running.</h3>'

@app.route('/meta')
def get_metadata():
    metadata = dict(
        SAMPLERATE=SAMPLERATE,
        DEVICE_NAME=DEVICE_NAME,
        NUM_CHANNELS=NUM_CHANNELS,
        BLOCKSIZE=BLOCKSIZE,
        NUM_BLOCKS_IN_BUFF=NUM_BLOCKS_IN_BUFF,
        BUFF_LEN=BUFF_LEN,
        TS_INIT=TS_INIT_FLASK,
        DATA_DIR=DATA_DIR,
        MAX_AMP_THRESH=MAX_AMP_THRESH,
        BUFF_DUR=BUFF_DUR
    )
    return jsonify(metadata)

@app.route('/state')
def get_capture_state():
    capture_status_info = load_json(CAPTURE_STATE_JSON_PATH)
    return jsonify(capture_status_info)

@app.route('/data', methods=['POST'])
def post_data():
    # unpack payload
    data = request.get_json(force=True)
    # print(data)
    max_amp = data['max_amp']
    ts_record = data['ts']
    wf_buff = np.array(data['wf'], dtype='int16')

    print([ts_record, max_amp, wf_buff[:5]])

    if 1:
        # data dir
        sess_data_dirname = os.path.join(DATA_DIR, str(TS_INIT_FLASK))
        if not os.path.exists(sess_data_dirname):
            os.makedirs(sess_data_dirname)

        #
        fname = str(ts_record) + '.wav'
        fpath = os.path.join(sess_data_dirname, fname)
        print(fpath)
        write_wav(fpath, SAMPLERATE, wf_buff, verbose=True)

        df = pd.DataFrame({
            'datetime': [convert_utc_to_dt(ts_record / 1e3)], # converts to local time zone
            "max_amp": [max_amp],
            "filepath": [fpath]
        })
        insert_or_update_db('raw', df, 'insert') # inserts dates after converting to UTC (since GMT tags are missing)

    res = dict(status='Success. Processed {} samples'.format(len(wf_buff)))
    return jsonify(res)


if __name__ == "__main__":
    FLASK_HOST = NET_CONFIG['HUB_IP']
    FLASK_PORT = NET_CONFIG['HUB_DATA_CAP_FLASK_PORT']
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
