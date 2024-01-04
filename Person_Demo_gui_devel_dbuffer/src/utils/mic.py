from collections import deque
from threading import Condition, Lock, Thread
import numpy as np
import librosa
from ctypes import *
import pyaudio
from matplotlib import cm
from PIL import Image

# Suppressing alsa error message
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
# Error handler do nothing
def py_error_handler(filename, line, function, err, fmt):
  pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
# Set error handler
asound.snd_lib_error_set_handler(c_error_handler)

class Mic:
    RATE = 16000
    CHUNK = 1024
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    def __init__(self, args):

        dev = pyaudio.PyAudio()
        stream = dev.open(
            format=Mic.FORMAT,
            channels=Mic.CHANNELS,
            rate=Mic.RATE,
            input=True,
            frames_per_buffer=Mic.CHUNK
        )

        num_samples = int(Mic.RATE / Mic.CHUNK * 5)

        self.dev = dev
        self.stream = stream
        self.data = deque(maxlen=num_samples)
        self.lock = Condition()

        self.norm_mean = np.ndarray([224, 224, 3], np.float32)
        self.norm_std = np.ndarray([224, 224, 3], np.float32)
        self.norm_mean[:, :] = np.array([0.485, 0.456, 0.406])
        self.norm_std[:, :] = 1 / np.array([0.229, 0.224, 0.225])

        for i in range(num_samples):
            continue
            self.data.append(self.stream.read(Mic.CHUNK))

        self.started        = False
        self.want_data      = False
        self.data_ready     = False
        self.thread         = Thread(target=self.update, args=())

    def start(self):
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if not self.want_data:
                self.data.append(self.stream.read(Mic.CHUNK))
            else:
                frame = []
                for i in self.data:
                    frame.extend(np.frombuffer(i, np.int16))

                data = librosa.feature.melspectrogram(
                    y=np.array(frame, dtype=np.float32),
                    sr=Mic.RATE,
                    hop_length=int(Mic.RATE * 0.015),
                    fmax=8000,
                    n_mels=224  # change to 256 if you see performance problems
                )
                data = librosa.power_to_db(data, ref=np.max)

                # Crop spectrogram
                h, w = data.shape
                xc, yc = w // 2, h // 2
                data = data[yc - 112:yc + 112, xc - 112:xc + 112]

                # Spectogram to Image
                sm = cm.ScalarMappable(cmap='viridis')
                data = sm.to_rgba(data, bytes=True)
                data = Image.fromarray(data)
                data = data.convert('RGB')
                data = np.array(data, dtype=np.float32)
                data *= (1 / 255.0)
                data -= self.norm_mean
                data *= self.norm_std
                data = data.transpose(2, 0, 1)
                data = np.ascontiguousarray(data, dtype=np.float32)

                with self.lock:
                    self.spectogram = data
                    self.data_ready = True
                    self.lock.wait()

    def data_available(self):
        if not self.data_ready:
            self.want_data = True
        return self.data_ready

    def read(self):
        with self.lock:
            image = np.array(self.spectogram)
            self.data_ready = False
            self.want_data = False
            self.lock.notify()
        return image

    def stop(self):
        if self.started:
            self.started = False
            self.thread.join()

    def __del__(self, **kwargs):
        self.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.dev.terminate()


if __name__ == "__main__":
    x = Mic()
    data = x.read()
    print(data)
