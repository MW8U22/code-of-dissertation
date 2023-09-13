import numpy as np
import wave
import matplotlib.pyplot as plt
import os

COOKED_DIR = 'D:/final pro/3444 Set 1 HW_NoHW/NOHW/'
SAVE_DIR = 'C:/Users/王民舟/Desktop/新建文件夹/'  # Change to a directory that you have write access to

plt.rcParams['figure.figsize'] = (4.48, 4.48)  # set figure_size
plt.rcParams['savefig.dpi'] = 50  # Image pixels and the output is 4.48*50=224

for root, dirs, files in os.walk(COOKED_DIR):
    print("Root =", root, "dirs =", dirs, "files =", files)
    for filename in files:
        if filename.endswith(".wav"):  # only solve.wav
            print(filename)
            path_one = os.path.join(root, filename)  # Concatenate directory path and file name to get the full path
            print(path_one)
            f = wave.open(path_one, 'rb')
            params = f.getparams()  # Return all audio parameters at one time, such as the number of channels, quantization bits, sampling frequency, and number of sampling points
            nchannels, sampwidth, framerate, nframes = params[:4]  # Channel/quantization number/sampling frequency/sampling points
            str_data = f.readframes(nframes)  # Specify the length to be read (in sampling points), and return string type data
            waveData = np.frombuffer(str_data, dtype=np.int16)  # Convert the string to int
            waveData = waveData * 1.0 / (max(abs(waveData)))  # Wave Amplitude normalization
            plt.specgram(waveData, NFFT=512, Fs=framerate, noverlap=500, scale_by_freq=True, sides='default')
            plt.ylabel('Frequency')
            plt.xlabel('Time(s)')
            plt.axis('off')
            name, ext = os.path.splitext(filename)  # Separate filename and extension
            save_path = os.path.join(SAVE_DIR, name + ".jpg")  # The path to save the generated image, using the original filename without the numeric index
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # The last two items are to remove white borders
            plt.close()  # Close the drawing window so a new image can be drawn




