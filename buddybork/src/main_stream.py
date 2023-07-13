

import time

from src.modules.audio_stream import AudioStreamer



aust = AudioStreamer()
aust.start()

while 1:
    time.sleep(60)
