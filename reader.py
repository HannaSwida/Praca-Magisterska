from pathlib import Path
import random
import numpy as np
import librosa


def load(path, num_samples):
    wav, _ = librosa.load(path, sr=16000)
    start = random.randint(0, wav.shape[0] - num_samples)
    return wav[start: (start + num_samples)]


class Reader:
    def __init__(self, data_path, batch_size=128, num_samples=3200):
        self.batch_size = batch_size
        self.num_samples = num_samples

        data_path = Path(data_path)

        speakers = {}

        for speaker_dir in data_path.iterdir():
            speaker_utterances = []

            for video_dir in speaker_dir.iterdir():
                for utterance_wav in video_dir.iterdir():
                    speaker_utterances.append(utterance_wav)

            speakers[speaker_dir.name] = speaker_utterances

        self.speakers = speakers

        self.speaker_list = list(sorted(speakers.keys()))

    def next(self):
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

    def speaker_to_id(self, speaker_name):
        return self.speaker_list.index(speaker_name)
