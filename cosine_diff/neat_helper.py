from model import CustomMLP, Encoder, SincConv
from voxceleb_loader import VCset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

n_batches = 1
batch_size = 128
n_workers = 1
input_dirs_file = 'source_vox1dir.txt'
encoder_path = "../folder_helper/checkpoints/MiModel//vox1_1000B_1500/1200vox1/encoder_141.pth"
classifier_path = "all_trained_1200labels.tar"
print(input_dirs_file)
tr_data = VCset(input_dirs_file, n_batches=n_batches, chunk_size=0.4)

tr_loader = DataLoader(tr_data, batch_size=len(tr_data), pin_memory=True, num_workers=n_workers)

checkpoint = torch.load(classifier_path, map_location=torch.device('cpu'))
