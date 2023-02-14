from pathlib import Path
import random
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
def load(path, num_samples=3200, chunk_size=0.2, overlap=0.01):
    wav, sr = librosa.load(path, sr=16000)
    chunk_samples = int(sr * chunk_size)
    overlap_samples = int(sr * overlap)
    print(num_samples)
    start = 0
    chunks = []
    while start + chunk_samples <= len(wav):
        chunks.append(wav[start:start+chunk_samples])
        start += chunk_samples - overlap_samples
    return chunks[:num_samples // chunk_samples]

class VoxCelebLoader(Dataset):
    def __init__(self, data_path, batch_size=128, num_samples=3200, chunk_size=0.2, overlap=0.01):
        self.batch_size = batch_size
        self.num_samples = num_samples
        print(num_samples, "n")
        self.chunk_size = chunk_size
        self.overlap = overlap
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
        triple = [
            load(speaker1utt1, self.num_samples, self.chunk_size, self.overlap),
            load(speaker1utt2, self.num_samples, self.chunk_size, self.overlap),
            load(speaker2utt1, self.num_samples, self.chunk_size, self.overlap)
        ]
        speaker_labels = [
            self.speaker_to_id(speaker1),
            self.speaker_to_id(speaker1),
            self.speaker_to_id(speaker2)
        ]
        # print("speaker_labels",speaker_labels)
        print(triple, speaker_labels)
        return triple, speaker_labels

    def __len__(self):
        return 12800


def make_batch(items):
    samples = [item[0] for item in items]
    speakers = [item[1] for item in items]

    return np.array(samples), np.array(speakers)