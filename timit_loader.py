from pathlib import Path
import random
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


def load(path, num_samples):
    wav, sr = librosa.load(path, sr=16000)
    #print(wav.shape)
    librosa.display.waveplot(wav, sr)
    S = np.abs(librosa.stft(wav))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()
    start = random.randint(0, wav.shape[0] - num_samples)
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
    return wav[start: (start + num_samples)]


class TimitLoader(Dataset):
    def __init__(self, data_path, batch_size=128, num_samples=3200):
        self.batch_size = batch_size
        self.num_samples = num_samples

        data_path = Path(data_path)
        speakers = {}

        for speaker_dir in data_path.iterdir():
            speaker_utterances = []

            for utterance_wav in speaker_dir.iterdir():
                if utterance_wav.name.lower().endswith(".wav"):
                    speaker_utterances.append(utterance_wav)

            if len(speaker_utterances) > 2:
                speakers[speaker_dir.name] = speaker_utterances

        self.speakers = speakers
        self.speaker_list = list(sorted(speakers.keys()))

    def speaker_to_id(self, speaker_name):
        return self.speaker_list.index(speaker_name)

    def __getitem__(self, _):
        # (3 utterances, samples), (3 speaker labels)
        [speaker1, speaker2] = random.sample(list(self.speakers.keys()), 2)
        [speaker1utt1, speaker1utt2] = random.sample(self.speakers[speaker1], 2)
        speaker2utt1 = random.choice(self.speakers[speaker2])

        triple = [
            load(speaker1utt1, self.num_samples),
            load(speaker1utt2, self.num_samples),
            load(speaker2utt1, self.num_samples)
        ]

        speaker_labels = [
            self.speaker_to_id(speaker1),
            self.speaker_to_id(speaker1),
            self.speaker_to_id(speaker2)
        ]
        return triple, speaker_labels

    def __len__(self):
        return 12800



class TimitLoaderDvector(Dataset):
    def __init__(self, data_path, batch_size=128, num_samples=3200, num_chunks=5): #numchunks
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_chunks = num_chunks

        data_path = Path(data_path)
        speakers = {}

        for speaker_dir in data_path.iterdir():
            speaker_utterances = []

            for utterance_wav in speaker_dir.iterdir():
                if utterance_wav.name.lower().endswith(".wav"):
                    speaker_utterances.append(utterance_wav)

            if len(speaker_utterances) > 2:
                speakers[speaker_dir.name] = speaker_utterances

        self.speakers = speakers
        self.speaker_list = list(sorted(speakers.keys()))
        print(self.speaker_list.__sizeof__())

    def speaker_to_id(self, speaker_name):
        return self.speaker_list.index(speaker_name)

    def __getitem__(self, i):
        speaker1 = list(self.speakers.keys())[i]
        utts = random.sample(self.speakers[speaker1], self.num_chunks)
        speaker_utts = [
            load(utt, self.num_samples) for utt in utts
        ]

        return torch.tensor(speaker_utts), speaker1 #self.speaker_to_id(speaker1)

    def __len__(self):
        return len(self.speakers)
