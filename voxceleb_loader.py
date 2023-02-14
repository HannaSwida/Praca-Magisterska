from pathlib import Path
import random
import librosa, librosa.display
from torch.utils.data import Dataset, DataLoader

def load(path, sample_length):
	#sr - sample rate
	wav, sr = librosa.load(path, sr=16000)
	start = random.randint(0, wav.shape[0] - sample_length)

	# print(wav.shape[0])
	return wav[start: (start + sample_length)]

class VoxCelebLoader(Dataset):
	def __init__(self, data_path, batch_size=128):
		self.batch_size = batch_size
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
		sample_length = 3200

		# (3 utterances, samples), (3 speaker labels)
		[speaker1, speaker2] = random.sample(list(self.speakers.keys()), 2)
		[speaker1utt1, speaker1utt2] = random.sample(self.speakers[speaker1], 2)
		speaker2utt1 = random.choice(self.speakers[speaker2])

		#To są chunks (c_1, c_2, c_rnd) 
		triple = [
			load(speaker1utt1, sample_length),
			load(speaker1utt2, sample_length),
			load(speaker2utt1, sample_length)
		]
		speaker_labels = [
			self.speaker_to_id(speaker1),
			self.speaker_to_id(speaker1),
			self.speaker_to_id(speaker2)
		]
		# print("speaker_labels",speaker_labels)
		return triple, speaker_labels

	def __len__(self):
		# batchsize * liczba iteracji
		return 12800


#def make_batch(items):
#    samples = [item[0] for item in items]
#    speakers = [item[1] for item in items]
#
#    return np.array(samples), np.array(speakers)
