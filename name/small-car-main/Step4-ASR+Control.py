import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests


asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

class ASR:
    def __init__(self):
        self._recognizer = OfflineRecognizer()
        raise NotImplementedError

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate=16000) -> str:
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=sample_rate)
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return s.result.text


class Whisper(ASR):
    def __init__(self, encoder_path: str, decoder_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )


class Paraformer(ASR):
    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )

print('正在加载模型...')
asr = Paraformer(
    model_path=f'{asr_path}/model.int8.onnx',
    tokens_path=f'{asr_path}/tokens.txt',
    # provider='cuda',
)
print('模型加载完成')

sample_rate = 16000

from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector
config = VadModelConfig(
    SileroVadModelConfig(
        model=f'{vad_path}/silero_vad.onnx',
        min_silence_duration=0.25,
    ),
    sample_rate=sample_rate
)
window_size = config.silero_vad.window_size
vad = VoiceActivityDetector(config, buffer_size_in_seconds=100)
samples_per_read = int(0.1 * sample_rate)

control_url = "http://192.168.192.123:5000/control"  

def send_command(text):
    global control_url
    ##TODO
    if '左' in text:
        response = requests.post(control_url, json={'command': "LEFT"})
    elif '右' in text:
        response = requests.post(control_url, json={'command': "RIGHT"})
    elif '前' in text:
        response = requests.post(control_url, json={'command': "FORWARD"})
    elif '停' in text:
        response = requests.post(control_url, json={'command': "STOP"})
    else:
        response = requests.post(control_url, json={'command': "STOP"})
    ##TODO

print('正在识别音频...')
idx = 1
buffer = []
try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                text = asr.transcribe(vad.front.samples, sample_rate=sample_rate)

                vad.pop()
                if len(text):
                    print()
                    print(f'第{idx}句：{text}')
                    send_command(text)
                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')