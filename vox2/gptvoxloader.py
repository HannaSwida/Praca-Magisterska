import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SpeakerTripletDataset(Dataset):
    def __init__(self, dataset_path, chunk_size, overlap, transform=None):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.transform = transform
        self.data = self._load_dataset()

    def _load_sample_file(self, filepath):
        y, sr = librosa.load(filepath,  sr=16000)
        chunk_samples = int(sr * self.chunk_size)
        overlap_samples = int(sr * self.overlap)

        start = 0
        while start < len(y):
            end = start + chunk_samples
            if end >= len(y):
                end = len(y)
            print(y[start:end].shape)
            yield y[start:end]
            start += chunk_samples - overlap_samples

    def _load_dataset(self):
        data = []
        print("fssssssfs")
        for speaker_folder in os.listdir(self.dataset_path):
            speaker_path = os.path.join(self.dataset_path, speaker_folder)
            for vid_folder in os.listdir(speaker_path):
                vid_path = os.path.join(speaker_path, vid_folder)
                for file_name in os.listdir(vid_path):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(vid_path, file_name)
                        chunks = list(self._load_sample_file(file_path))
                        if len(chunks) >= 3:
                            # Get three random chunks from the same speaker
                            anchor, positive, negative = np.random.choice(chunks, 3, replace=False)
                            data.append((anchor, positive, negative, speaker_folder))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor, positive, negative, speaker = self.data[idx]
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative, speaker


# Define the dataset path, chunk size and overlap
dataset_path = 'dev/vox1selected/'
chunk_size = 200  # in ms
overlap = 10  # in ms

# Initialize the dataset and the data loader
dataset = SpeakerTripletDataset(dataset_path, chunk_size, overlap)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
print(data_loader)