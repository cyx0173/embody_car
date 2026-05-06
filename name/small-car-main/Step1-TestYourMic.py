import sounddevice as sd
import numpy as np
import sys


devices = sd.query_devices()
default_input_device_idx = sd.default.device[0]
default_output_device_idx = sd.default.device[1]
print(f'使用默认声音输入设备：{devices[default_input_device_idx]["name"]}')
print(f'使用默认声音输出设备：{devices[default_output_device_idx]["name"]}')
sample_rate = 16000
samples_per_read = int(0.1 * sample_rate)
print('准备开始录音5秒...')
print('倒计时：', end='')
for i in range(3):
    print('%d .. ' % (3 - i), end='')
    sys.stdout.flush()
    sd.sleep(1000)
print('\n开始录音...')
with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
    pcm = np.array([], dtype=np.float32)
    for i in range(50):
        samples, _ = s.read(samples_per_read)
        samples = samples.reshape(-1)
        pcm = np.concatenate((pcm, samples), axis=0)
print('录音结束\n')
sd.sleep(1000)
print('正在回放录音...')
sd.play(pcm, samplerate=sample_rate)
sd.wait()
print('回放结束')
