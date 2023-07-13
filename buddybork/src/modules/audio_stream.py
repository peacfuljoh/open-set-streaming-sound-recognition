'''Audio stream class for continuous data capture and store in database'''

import os
from queue import Queue
from threading import Thread
import time

import numpy as np
import pandas as pd
import sounddevice as sd

from src.constants_stream import SAMPLERATE, DEVICE_NAME, NUM_CHANNELS, BLOCKSIZE, NUM_BLOCKS_IN_BUFF, BUFF_LEN, \
    TS_INIT, DATA_DIR, MAX_AMP_THRESH, BUFF_DUR, CAPTURE_STATE_JSON_PATH
from ossr_utils.misc_utils import get_ts_now, convert_utc_to_dt, get_seg_amp_metric
from src.utils.db_utils import insert_or_update_db
from ossr_utils.io_utils import load_json
from ossr_utils.audio_utils import write_wav


# print(sd.query_devices())



class AudioStreamer():
    def __init__(self):
        self._stream = None
        self._sess_data_dirname = os.path.join(DATA_DIR, str(TS_INIT))

        self._q_in = Queue()

        self._wf_buff = np.zeros(BUFF_LEN, dtype='int16')

        self._running = False # don't init data collection without explicit signal from controller
        print('Data collection is OFF')

    def start(self):
        # init stream
        self._stream = sd.InputStream(
            samplerate=SAMPLERATE,
            device=DEVICE_NAME,
            channels=NUM_CHANNELS,
            callback=self._stream_to_q,
            blocksize=BLOCKSIZE,
            dtype='int16'
        )
        # self._stream.start()

        # init data capture and process thread
        self._process_thread = Thread(target=self._process, daemon=True)
        self._process_thread.start()

        # init status check thread
        self._status_check_thread = Thread(target=self._status_check, daemon=True)
        self._status_check_thread.start()

    def _stream_to_q(self, indata, frames, time, status):
        """This is called for each audio block."""
        self._q_in.put(indata[:, 0].copy())

    def _status_check(self):
        while 1:
            # determine run status
            capture_status_info = load_json(CAPTURE_STATE_JSON_PATH)
            run_status = capture_status_info['capture_state']

            # toggle if necessary
            if self._running and not run_status: # start stream
                self._running = False
                self._stream.stop()
                self._q_in.put('reset')
                print('Data collection is OFF')
            elif not self._running and run_status: # stop stream
                while self._q_in.qsize(): # clear the queue
                    self._q_in.get()
                self._running = True
                self._stream.start()
                print('Data collection is ON')

            time.sleep(BUFF_DUR)

    def _process(self):
        os.makedirs(self._sess_data_dirname)

        idx_block = 0

        while 1:
            if not self._running:
                idx_block = 0
                time.sleep(BUFF_DUR)
                continue

            item_ = self._q_in.get()

            if isinstance(item_, str) and item_ == 'reset':
                idx_block = 0
                continue

            wf = item_

            if len(wf) != BLOCKSIZE:
                print('Warning: Acquired block has {} samples, expected {}'.format(len(wf), BLOCKSIZE))
                continue

            self._wf_buff[idx_block * BLOCKSIZE:(idx_block + 1) * BLOCKSIZE] = wf

            idx_block += 1

            if idx_block == NUM_BLOCKS_IN_BUFF:
                idx_block = 0

                # write to file and db
                max_amp = get_seg_amp_metric(self._wf_buff)
                if max_amp > MAX_AMP_THRESH:
                    self._save_buff_info(max_amp)
                else:
                    print('Skipping segment. max_amp = {} < {}'.format(max_amp, MAX_AMP_THRESH))

    def _save_buff_info(self, max_amp):
        print('max_amp: ' + str(max_amp))

        ts_record = get_ts_now() # ms, UTC

        fname = str(ts_record) + '.wav'
        fpath = os.path.join(self._sess_data_dirname, fname)

        write_wav(fpath, SAMPLERATE, self._wf_buff, verbose=True)

        df = pd.DataFrame({
            'datetime': [convert_utc_to_dt(ts_record / 1e3)], # converts to local time zone
            "max_amp": [max_amp],
            "filepath": [fpath]
        })
        insert_or_update_db('raw', df, 'insert') # inserts dates after converting to UTC (since GMT tags are missing)
