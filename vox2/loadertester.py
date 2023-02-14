#!/usr/bin/env python3.8
#pp.py ./main.py --data .\training-data\timit --loader timit
from pathlib import Path

from vox1origimat import VoxCelebLoader
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Praca magisterska')
parser.add_argument('--loader', default='voxceleb',
                    help='dataset type')
parser.add_argument('--data', default='vox2/dev/vox1selected',
                    help='dataset name')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# pp.py main.py --lr 0.01

def make_batch(items):
    samples = [item[0] for item in items]
    speakers = [item[1] for item in items]

    return np.array(samples), np.array(speakers)


def main():

    args = parser.parse_args()

    data_path = Path('dev/vox1selected')
    print("afff")
    for speaker_dir in data_path.iterdir():
        print("a")
        print(speaker_dir)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Using {}".format(device))

    Loader = ({
        "voxceleb": VoxCelebLoader,
    })[args.loader]

    voices_loader = Loader(args.data)

    train_loader = DataLoader(dataset=voices_loader,
                              shuffle=True,
                              num_workers=2,
                              batch_size=128,
                              collate_fn=make_batch)



if __name__ == "__main__":
    main()
