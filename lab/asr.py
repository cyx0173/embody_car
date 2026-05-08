"""
语音识别模块
使用阿里云 DashScope Gummy 进行一句话语音识别。
"""

from __future__ import annotations

import os
import time

import dashscope
import pyaudio
from dashscope.audio.asr import TranslationRecognizerCallback, TranslationRecognizerChat


QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-029be25ee95448a18ac98cbc2d89b12d")


def list_input_devices() -> list[dict]:
    p = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info.get("name", ""),
                        "channels": int(info.get("maxInputChannels", 0)),
                        "sample_rate": int(float(info.get("defaultSampleRate", 0))),
                    }
                )
    finally:
        p.terminate()
    return devices


class QwenASR:
    def __init__(
        self,
        input_device_index: int | None = None,
        sample_rate: int = 16000,
        max_end_silence: int = 1000,
        timeout_seconds: float = 12.0,
    ):
        dashscope.api_key = QWEN_API_KEY
        self.mic = None
        self.stream = None
        self.final_text = ""
        self.sample_rate = sample_rate
        self.max_end_silence = max_end_silence
        self.timeout_seconds = timeout_seconds
        self.input_device_index = input_device_index if input_device_index is not None else self._find_input_device()
        self._completed = False
        self._error = None

    def _find_input_device(self) -> int | None:
        devices = list_input_devices()
        for dev in devices:
            name = dev["name"]
            if "USB" in name or "Camera" in name or "麦克风" in name or "Microphone" in name:
                print(f"[ASR] 使用音频设备: {name} (index={dev['index']})")
                return dev["index"]
        if devices:
            dev = devices[0]
            print(f"[ASR] 使用默认可用输入设备: {dev['name']} (index={dev['index']})")
            return dev["index"]
        print("[ASR] 未找到输入设备，将尝试系统默认输入。")
        return None

    def listen(self, voice_status=None) -> str:
        """阻塞式监听一句话，检测到句尾或超时后返回转写文本。"""
        self.final_text = ""
        self._completed = False
        self._error = None
        if voice_status is None:
            voice_status = {}

        callback = TranslationRecognizerCallback()
        callback.on_open = self._on_open
        callback.on_close = self._on_close
        callback.on_complete = self._on_complete
        callback.on_error = self._on_error
        callback.on_event = self._on_event

        translator = TranslationRecognizerChat(
            model="gummy-chat-v1",
            format="pcm",
            sample_rate=self.sample_rate,
            transcription_enabled=True,
            translation_enabled=False,
            max_end_silence=self.max_end_silence,
            callback=callback,
        )

        print("[ASR] 正在连接语音识别服务...", flush=True)
        start = time.monotonic()
        try:
            translator.start()
            print("[ASR] 已开始监听，请说话...", flush=True)
            while voice_status.get("active", True):
                if voice_status.get("is_speaking", False):
                    break
                if self._completed:
                    break
                if time.monotonic() - start > self.timeout_seconds:
                    print("[ASR] 本轮监听超时。", flush=True)
                    break
                if not self.stream or self.stream.is_stopped():
                    time.sleep(0.05)
                    continue

                data = self.stream.read(3200, exception_on_overflow=False)
                if not translator.send_audio_frame(data):
                    break
        except Exception as e:
            print(f"[ASR] 识别失败: {e}", flush=True)
        finally:
            try:
                if getattr(translator, "_running", False):
                    translator.stop()
            except Exception:
                pass
            self._close_audio()

        text = self.final_text.strip()
        if text:
            print(f"[ASR] 识别结果: {text}", flush=True)
        time.sleep(0.2)
        return text

    def _on_open(self):
        self.mic = pyaudio.PyAudio()
        kwargs = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": self.sample_rate,
            "input": True,
            "frames_per_buffer": 1024,
        }
        if self.input_device_index is not None:
            kwargs["input_device_index"] = self.input_device_index
        try:
            self.stream = self.mic.open(**kwargs)
            print("[ASR] 麦克风已打开。", flush=True)
        except Exception as e:
            print(f"[ASR] 指定音频设备打开失败: {e}，改用系统默认输入。", flush=True)
            kwargs.pop("input_device_index", None)
            self.stream = self.mic.open(**kwargs)
            print("[ASR] 麦克风已打开。", flush=True)

    def _on_close(self):
        pass

    def _on_complete(self):
        self._completed = True

    def _on_error(self, message):
        self._error = message
        self._completed = True
        print(f"[ASR] 服务端错误: {message}", flush=True)

    def _on_event(self, request_id, transcription_result, translation_result, usage):
        if transcription_result and transcription_result.text:
            self.final_text = transcription_result.text
            if getattr(transcription_result, "is_sentence_end", False):
                self._completed = True

    def _close_audio(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()
        self.stream = None
        self.mic = None


if __name__ == "__main__":
    asr = QwenASR()
    text = asr.listen()
    print(f"识别结果: {text}")
