from pathlib import Path
import random
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from constants import BATCHES_PER_EPOCH


def load(path, num_samples):
    wav, sr = librosa.load(path, sr=16000)
    # print(wav.shape)
    # librosa.display.waveplot(wav, sr)
    # plt.show()
    start = random.randint(0, wav.shape[0] - num_samples)
    return wav[start: (start + num_samples)]


class VoxCelebLoader(Dataset):
    def __init__(self, data_path, batch_size=128, num_samples=3200):
        self.batch_size = batch_size
        self.num_samples = num_samples

        data_path = Path(data_path)

        speakers = {}

        for speaker_dir in data_path.iterdir():
            speaker_utterances = []

            for video_dir in speaker_dir.iterdir():
                for utterance_wav in video_dir.iterdir():
                    if utterance_wav.name.endswith(".wav"):
                        speaker_utterances.append(utterance_wav)

            if len(speaker_utterances) > 2:
                speakers[speaker_dir.name] = speaker_utterances

        self.speakers = speakers
        self.speaker_list = list(sorted(speakers.keys()))

    def speaker_to_id(self, speaker_name):
        return self.speaker_list.index(speaker_name)

    def __getitem__(self, _):
        # (batch size, 3 utterances, samples)
        batch = []
        speakers = []

        for i in range(self.batch_size):
            [speaker1, speaker2] = random.sample(list(self.speakers.keys()), 2)
            [speaker1utt1, speaker1utt2] = random.sample(self.speakers[speaker1], 2)
            speaker2utt1 = random.choice(self.speakers[speaker2])

            batch.append([
                load(speaker1utt1, self.num_samples),
                load(speaker1utt2, self.num_samples),
                load(speaker2utt1, self.num_samples)
            ])

            speakers.append([
                self.speaker_to_id(speaker1),
                self.speaker_to_id(speaker1),
                self.speaker_to_id(speaker2)
            ])
        return np.array(batch), np.array(speakers)

    def __len__(self):
        return BATCHES_PER_EPOCH
