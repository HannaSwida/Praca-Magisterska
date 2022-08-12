from pathlib import Path
import random
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import torch
import subprocess
import sys
import numpy
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
wav, sr = librosa.load("testing-data/timit/FADG0/00385.m4a", sr=16000)
librosa.display.waveplot(wav, sr)
S = np.abs(librosa.stft(wav))
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                   ref=np.max),
                           y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()
n_fft = 2048
S = librosa.stft(wav, n_fft=n_fft, hop_length=n_fft // 2)
# convert to db
# (for your CNN you might want to skip this and rather ensure zero mean and unit variance)
D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
# average over file
D_AVG = np.mean(D, axis=1)

plt.bar(np.arange(D_AVG.shape[0]), D_AVG)
x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
x_ticks_labels = [str(sr / 2048 * n) + 'Hz' for n in x_ticks_positions]
plt.xticks(x_ticks_positions, x_ticks_labels)
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.show()