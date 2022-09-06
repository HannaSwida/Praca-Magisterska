#!/usr/bin/env python3.8
#python ./main.py --data .\training-data\timit --loader timit
from vox_celeb_loader import VoxCelebLoader
from model import Model
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as Fun
import torch.optim as optim
import constants as consts
import argparse
import os
import numpy as np
from constants import BATCHES_PER_EPOCH, BATCHES

parser = argparse.ArgumentParser(description='Praca magisterska')
parser.add_argument('--loader', default='voxceleb',
                    help='dataset type')
parser.add_argument('--data', default='./training-data/timit',
                    help='dataset name')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


# python main.py --lr 0.01


def make_batch(items):
    samples = [item[0] for item in items]
    speakers = [item[1] for item in items]
    print(len(speakers[0]))
    return np.array(samples), np.array(speakers)


def main():
    args = parser.parse_args()

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Using {}".format(device))

    Loader = ({
        "voxceleb": VoxCelebLoader,
        #"timit": TimitLoader,
    })[args.loader]
    
    voices_loader = Loader(args.data)

    train_loader = DataLoader(dataset=voices_loader,
                              shuffle=True,
                              num_workers=2,
                              batch_size=BATCHES,
                              collate_fn=make_batch)

    model = Model(168)
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=consts.alpha, eps=1e-07)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading given checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded given checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    def save_checkpoint(state, filename):
        torch.save(state, filename)

    criterion = torch.nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    print(len(voices_loader))

    with open("loss.txt", "a") as lossFile:
        model.train()
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            loss_sum= 0
            lossFile.write("Epoch {}/{}:".format(epoch + 1, args.start_epoch + args.epochs) + "\n")
            print("Epoch {}/{}:".format(epoch + 1, args.start_epoch + args.epochs))
            for i, (batch, speakers) in enumerate(train_loader):
                if i >= BATCHES_PER_EPOCH:
                    break
                print("Batch {}/{}   ".format(i + 1, BATCHES_PER_EPOCH))
                optimizer.zero_grad()
                score_posp, score_negp, speakers_probs, speakers = model(torch.tensor(batch, device=device), torch.tensor(speakers, dtype=torch.long, device=device))

                loss = loss_fn(score_negp, score_posp, speakers, speakers_probs)
                loss_sum = loss_sum + loss
                loss.mean().backward()
                print(loss.mean())
                optimizer.step()
            print("Mean loss:", loss_sum/BATCHES_PER_EPOCH)
            lossFile.write(str(loss_sum/BATCHES_PER_EPOCH) + "\n")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename="./checkpoints_{}/checkpoint_test{}.pth.tar".format(args.loader, epoch))
            scheduler.step(loss)

def loss_fn(score_negp, score_posp, speakers, speakers_probs):

    ## próba z 27.03.2022. Wzór nr (3) z https://arxiv.org/pdf/1812.00271.pdf
    #print(torch.log(1-score_negp))
    #print(score_negp)
    return (-torch.mean(torch.log(score_posp))) - torch.mean(torch.log(1-score_negp))


if __name__ == "__main__":
    main()
