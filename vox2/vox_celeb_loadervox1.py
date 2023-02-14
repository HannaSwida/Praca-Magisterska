from pathlib import Path
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import random


def load(path, num_samples):
    wav, sr = librosa.load(path, sr=16000)
    return wav


class VoxCelebLoader(Dataset):
    def __init__(self, data_path, batch_size=128, num_samples=3200):
        self.batch_size = batch_size
        self.num_samples = num_samples
        print(num_samples)
        self.sample_rate = 16000
        self.window_length = int(0.01 * self.sample_rate)
        self.window_shift = int(0.02 * self.sample_rate)
        data_path = Path(data_path)
        speakers = {}
        for speaker_dir in data_path.iterdir():
            speaker_utterances = []
            for video_dir in speaker_dir.iterdir():
                for utterance_wav in video_dir.iterdir():
                    if utterance_wav.name.endswith(".m4a"):
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
        print("loading 123")
        samples1 = load(speaker1utt1, self.num_samples)
        samples2 = load(speaker1utt2, self.num_samples)
        samples3 = load(speaker2utt1, self.num_samples)
        print("end loading 123")

        num_windows1 = (len(samples1) - self.window_length) // self.window_shift + 1
        num_windows2 = (len(samples2) - self.window_length) // self.window_shift + 1
        num_windows3 = (len(samples3) - self.window_length) // self.window_shift + 1

        for i in range(num_windows1):
            start = i * self.window_shift
            end = start + self.window_length
            triple = [
                samples1[start:end],
                samples2[start:end],
                samples3[start:end]
            ]
            speaker_labels = [
                self.speaker_to_id(speaker1),
                self.speaker_to_id(speaker1),
                self.speaker_to_id(speaker2)
                ]
            yield (triple, speaker_labels)

    def __len__(self):
        return sum(len(utterances) for utterances in self.speakers.values()) // self.batch_size

data_path = "dev/vox1selected/"
voxceleb = VoxCelebLoader(data_path)
dataloader = DataLoader(voxceleb, batch_size=32, shuffle=True)

for i, (triple, labels) in enumerate(dataloader):
    samples = triple
    speaker_labels = labels
    if i == 0:
        break

print("Samples Shape:", samples.shape)
print("Speaker Labels Shape:", speaker_labels.shape)