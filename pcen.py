import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# 音频文件路径
audio_path = "D:\\final pro\\3444 Set 1 HW_NoHW\\HW\\"
file_list = [f for f in os.listdir(audio_path) if f.endswith('.wav')]

# 保存的目标路径
save_path = "C:\\Users\\王民舟\\Desktop\\final project\\鲸鱼检索\\鲸鱼检索-3444\\Data\\pcen_3444\\HW\\"

def pcen(S, s=0.025, alpha=0.98, delta=2, r=0.5, eps=1e-6):
    # Manual implementation of PCEN
    M = np.maximum(S, eps)
    smooth = (1 - s) * M + s * M
    return (S / (eps + smooth**alpha + delta))**r


for file in file_list:
    full_path = os.path.join(audio_path, file)

    # 读取音频文件
    y, sr = sf.read(full_path)

    # 计算频谱图
    f, t, S = spectrogram(y, fs=sr, window='hann', nperseg=1024, noverlap=768, nfft=4096)

    # 先应用PCEN
    S_PCEN = pcen(np.abs(S))
    # 转换到dB尺度
    S_PCEN_dB = 20 * np.log10(S_PCEN + 1e-6)

    # 创建并保存PCEN的频谱图
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, np.log10(1 + f / 700), S_PCEN_dB, shading='gouraud', cmap='viridis')
    plt.colorbar()
    plt.clim(-5, 20)
    plt.axis('off')
    plt.title(f'PCEN 频谱图 - {file}')
    plt.tight_layout()

    # 定义保存的文件名和路径
    save_filename = os.path.splitext(file)[0] + "_pcen.png"
    save_filepath = os.path.join(save_path, save_filename)

    # 保存图像
    plt.savefig(save_filepath, bbox_inches='tight', pad_inches=0)
    plt.close()
