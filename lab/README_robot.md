# Robot Agent Pipeline

## 当前链路

入口文件是 `main.py`，现在串起了：

1. 语音或文本输入：`asr.QwenASR`，也可用 `--text` 或 `--once` 跳过麦克风。
2. 意图解析：`nlu.NLU`，把自然语言分成视觉问答、目标追踪、机械臂交互。
3. 图像采集：`perception.Camera`，通过 OpenCV 读取摄像头。
4. YOLO 感知：`perception.YoloPerception`，输出检测标签、置信度和中心点。
5. 视觉问答：`visual_qa.VisualQA` 调用 `vlm.VLM`，把图像和 YOLO 上下文交给 GLM-4V 生成回复。
6. 回复：默认使用 `tts.TTS`，如果本地 Qwen TTS 模型不存在会退回 macOS `say`；调试时可用 `--no-tts` 只打印。
7. 动作接口：`robotic_interaction.RoboticInteraction` 和 `visual_tracking.VisualTracking`，默认 dry-run，不会驱动机械臂；加 `--execute` 后才实际调用硬件。

## 还缺的关键环节

- 精确抓取还缺手眼标定结果和像素坐标到机械臂坐标的转换。
- 还缺末端执行器的开合控制接口，所以现在只能做到识别目标、追踪、复位和动作意图确认。
- YOLO 的类别是英文 COCO 标签，中文目标词需要在 `nlu.py` 的 `OBJECT_ALIASES` 里继续补。
- ASR 和 VLM 依赖云 API，建议用环境变量配置 `DASHSCOPE_API_KEY` 和 `ZHIPUAI_API_KEY`。
- 本地 Qwen TTS 模型目录 `Qwen3-TTS-0.6B-CustomVoice` 当前不在仓库里；没有它时会自动用系统 `say`。

## 环境

已创建 conda 环境：

```bash
conda activate robot
```

已安装核心依赖：`opencv-python`、`ultralytics`、`zhipuai`、`dashscope`、`pyaudio`、`sounddevice`、`pyserial`、`torch`。

如果终端前面已经显示 `(robot)`，下面命令直接用 `python`，不要再套 `conda run`；交互式语音程序用 `conda run` 时可能会缓冲输出，看起来像“没反应”。

## 推荐运行方式

先用文本和无 TTS 调试视觉问答：

```bash
python lab/main.py --once "我正在电脑上做什么" --no-tts --show-camera
```

进入文本交互模式：

```bash
python lab/main.py --text --no-tts --show-camera
```

查看麦克风设备：

```bash
python lab/main.py --list-audio-devices
```

先确认摄像头窗口和 YOLO 框正常：

```bash
python lab/main.py --preview-camera
```

启用语音输入和语音播报，但仍不驱动机械臂：

```bash
python lab/main.py --asr-device 2 --show-camera
```

如果有多个麦克风，指定输入设备：

```bash
python lab/main.py --asr-device 0 --show-camera
```

如果一句话经常截断，可以把句尾静音调长；如果等待太久，可以把超时调短：

```bash
python lab/main.py --asr-device 2 --show-camera --max-end-silence 1500 --asr-timeout 8
```

确认机械臂串口、摄像头和安全边界无误后，再启用真实执行：

```bash
python lab/main.py --execute --arm-port /dev/cu.usbmodem5AE60562991 --show-camera
```
