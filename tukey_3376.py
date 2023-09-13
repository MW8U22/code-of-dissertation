import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft


def cqt(y, sr=44100, hop_size=512, bins_per_octave=12, n_bins=7 * 12, f_min=130.81):
    # Prepare CQT kernel
    fft_len = 2 ** int(np.ceil(np.log2(hop_size + max(4096, hop_size))))
    freqs = f_min * 2 ** (np.arange(n_bins, dtype=np.float32) / bins_per_octave)
    kern = np.empty((n_bins, fft_len), dtype=np.complex64)

    for i, f in enumerate(freqs):
        kern[i] = np.exp(2j * np.pi * f * np.arange(fft_len) / sr)
        kern[i] /= np.linalg.norm(kern[i], ord=2)

    # Compute CQT
    hop_size = int(hop_size)
    y_pad = np.pad(y, int(fft_len // 2), mode='reflect')

    frames = np.array([y_pad[i:i + fft_len] for i in range(0, len(y_pad) - fft_len + 1, hop_size)])
    cqt_result = fft(frames) @ kern.T
    return 20 * np.log10(np.abs(cqt_result) + 1e-6)


COOKED_DIR = 'C:/Users/王民舟/Desktop/'
SAVE_DIR = 'C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3376/'

plt.rcParams['figure.figsize'] = (4.48, 4.48)
plt.rcParams['savefig.dpi'] = 50

for root, dirs, files in os.walk(COOKED_DIR):
    for filename in files:
        if filename.endswith(".wav"):
            path_one = os.path.join(root, filename)

            # Using scipy to read wav file
            sr, y = wavfile.read(path_one)
            y = y.astype(np.float32) / np.iinfo(y.dtype).max  # Normalize

            # Compute CQT
            C = cqt(y, sr=sr)

            plt.imshow(C.T, aspect='auto', cmap='inferno', origin='lower')
            plt.axis('off')

            name, ext = os.path.splitext(filename)
            save_path = os.path.join(SAVE_DIR, name + ".jpg")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()



