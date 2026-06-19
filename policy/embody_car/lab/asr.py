"""
语音识别模块
使用阿里云DashScope的Gummy语音识别
"""

import time

try:
    import pyaudio
    import dashscope
    from dashscope.audio.asr import TranslationRecognizerCallback, TranslationRecognizerChat
except ImportError as exc:
    pyaudio = None
    dashscope = None
    TranslationRecognizerCallback = None
    TranslationRecognizerChat = None
    ASR_IMPORT_ERROR = exc
else:
    ASR_IMPORT_ERROR = None

QWEN_API_KEY = "sk-029be25ee95448a18ac98cbc2d89b12d"

class QwenASR:
    def __init__(self):
        self.text_fallback = ASR_IMPORT_ERROR is not None
        if dashscope is not None:
            dashscope.api_key = QWEN_API_KEY
        else:
            print(f"[ASR] 语音依赖不可用，改用命令行输入: {ASR_IMPORT_ERROR}")
        self.mic = None
        self.stream = None
        self.final_text = ""
        self.input_device_index = None if self.text_fallback else self._find_usb_mic_index()

    def _find_usb_mic_index(self):
        """自动查找包含 'USB' 或 'Camera' 的音频输入设备索引"""
        if pyaudio is None:
            return None
        p = pyaudio.PyAudio()
        target_idx = None
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            dev_name = dev_info.get('name', '')
            # 根据你的设备名称特征筛选：A4tech USB2.0 Camera
            if dev_info.get('maxInputChannels', 0) > 0:
                if 'USB' in dev_name or 'Camera' in dev_name:
                    target_idx = i
                    print(f"[ASR] 使用音频设备: {dev_name} (index={i})")
                    break
        p.terminate()
        if target_idx is None:
            print("[ASR] 警告：未找到USB麦克风，将使用系统默认输入设备")
        return target_idx

    def listen(self, voice_status=None):
        """阻塞式监听，返回转写的最终文字"""
        if self.text_fallback:
            return input("请输入指令> ").strip()

        self.final_text = ""
        if voice_status is None:
            voice_status = {}

        callback = TranslationRecognizerCallback()
        callback.on_open = self._on_open
        callback.on_close = self._on_close
        callback.on_event = self._on_event

        translator = TranslationRecognizerChat(
            model="gummy-chat-v1",
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            translation_enabled=False,
            max_end_silence=1000,
            callback=callback
        )

        translator.start()

        try:
            while True:
                if not voice_status.get('active', True):
                    break
                if voice_status.get('is_speaking', False):
                    break

                if self.stream and not self.stream.is_stopped():
                    try:
                        # 每次读取 3200 帧（对应 200ms @16kHz）
                        data = self.stream.read(3200, exception_on_overflow=False)
                        if not translator.send_audio_frame(data):
                            break
                    except OSError as e:
                        # ALSA 设备错误，可能是设备断开或驱动异常
                        print(f"[ASR] 音频读取错误: {e}")
                        break
                    except Exception as e:
                        print(f"[ASR] 未知错误: {e}")
                        break
                else:
                    break
        finally:
            translator.stop()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.mic:
                self.mic.terminate()
            self.stream = None
            self.mic = None

        time.sleep(0.3)  # 等待音频设备完全释放
        return self.final_text

    def _on_open(self):
        if pyaudio is None:
            return
        self.mic = pyaudio.PyAudio()
        # 关键修改：添加 frames_per_buffer 并指定设备索引
        try:
            stream_kwargs = {
                "format": pyaudio.paInt16,
                "channels": 1,
                "rate": 16000,
                "input": True,
                "frames_per_buffer": 1024,
            }
            if self.input_device_index is not None:
                stream_kwargs["input_device_index"] = self.input_device_index
            self.stream = self.mic.open(**stream_kwargs)
        except Exception as e:
            print(f"[ASR] 打开音频流失败: {e}")
            # 降级尝试：不指定设备索引，让系统选择默认设备
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

    def _on_close(self):
        pass

    def _on_event(self, request_id, transcription_result, translation_result, usage):
        if transcription_result and transcription_result.text:
            self.final_text = transcription_result.text
if __name__ == "__main__":
    asr = QwenASR()
    print("请说话（5秒无声音自动结束）...")
    text = asr.listen()
    print(f"识别结果: {text}")
