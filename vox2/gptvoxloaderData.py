import os
import librosa
import torch
from torch.utils.data import Dataset, DataLoader


class SpeakerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_sample_file(filepath, chunk_size=0.2, overlap=0.01):
    # Load the audio file
    y, sr = librosa.load(filepath)
    # Calculate the number of samples in each chunk and overlap
    chunk_samples = int(sr * chunk_size)
    overlap_samples = int(sr * overlap)

    # Split the audio into overlapping chunks
    start = 0
    while start < len(y):
        end = start + chunk_samples
        if end >= len(y):
            end = len(y)

        yield y[start:end]

        start += chunk_samples - overlap_samples


def load_dataset(dataset_path, chunk_size, overlap):
    data = []
    for speaker_folder in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker_folder)
        for vid_folder in os.listdir(speaker_path):
            vid_path = os.path.join(speaker_path, vid_folder)
            for file_name in os.listdir(vid_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(vid_path, file_name)
                    chunks = load_sample_file(file_path, chunk_size, overlap)
                    for chunk in chunks:
                        data.append((chunk, speaker_folder))
    print(data)
    return data


dataset_path = "dev/vox1selected/"
chunk_size = 0.2
overlap = 0.01
data = load_dataset(dataset_path, chunk_size, overlap)
dataset = SpeakerDataset(data)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

for i, batch in enumerate(dataloader):
    print("aaaa")
    if i == 0:
        break
    audio_chunks, speaker_labels = batch
    print(audio_chunks.shape)
    print(speaker_labels)