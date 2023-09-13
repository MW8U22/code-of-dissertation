import numpy as np
import matplotlib.pyplot as plt

# 定义Hann窗的长度
N = 512

# 生成Hann窗
hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

# 绘制Hann窗
plt.figure(figsize=(10, 4))
plt.plot(hann_window)
plt.title("Hann Window")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
