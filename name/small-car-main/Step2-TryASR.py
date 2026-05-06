import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import os
import time
import numpy as np

model_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'

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
paraformer = Paraformer(
    model_path=f'{model_path}/model.int8.onnx',
    tokens_path=f'{model_path}/tokens.txt',
    # provider='cuda',
)
print('模型加载完成')

print('正在识别测试音频...')
start_time = time.time()
for audio_file in os.listdir(f'{model_path}/test_wavs'):
    if audio_file.endswith('.wav'):
        audio_path = f'{model_path}/test_wavs/{audio_file}'
        print(f'音频文件：{audio_path}')
        pcm, sample_rate = librosa.load(audio_path, sr=16000)
        print(f'音频时长：{len(pcm) / sample_rate:.2f}秒')
        print('正在识别...')
        result = paraformer.transcribe(pcm, sample_rate=sample_rate)
        print(f'识别结果：{result}\n')
end_time = time.time()
print(f'全部识别完成，总耗时：{end_time - start_time:.2f}秒')
