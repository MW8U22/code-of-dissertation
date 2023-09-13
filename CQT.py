import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

COOKED_DIR = 'D:/final pro/Labels PACAP Set 1 3376/NOHW/'
SAVE_DIR = 'C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3376/data/CQT/NOHW/'

plt.rcParams['figure.figsize'] = (4.48, 4.48)
plt.rcParams['savefig.dpi'] = 50

def compute_librosa_cqt(y, sr):
    C = np.abs(librosa.cqt(y, sr=sr, n_bins=72, bins_per_octave=12))  # Adjust n_bins value
    return C

for root, dirs, files in os.walk(COOKED_DIR):
    for filename in files:
        if filename.endswith(".wav"):
            path_one = os.path.join(root, filename)
            y, sr = librosa.load(path_one, sr=None)
            cqt_result = compute_librosa_cqt(y, sr)

            librosa.display.specshow(librosa.amplitude_to_db(cqt_result, ref=np.max))
            plt.axis('off')
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(SAVE_DIR, name + ".jpg")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
