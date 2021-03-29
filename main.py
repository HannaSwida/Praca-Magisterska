from vox_celeb_loader import VoxCelebLoader
from model import Model
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import constants as consts
import argparse
import os

# TODO GPUCPU
parser = argparse.ArgumentParser(description='Praca magisterska')
parser.add_argument('data', metavar='DIR',
                    help='dataset name')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')  # todo
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


#python main.py --lr 0.01 Voxceleb


def main():
    args = parser.parse_args()

    if args.data == 'Voxceleb':
        voices_loader = VoxCelebLoader("training-data/voxceleb")

    train_loader = DataLoader(dataset=voices_loader,
                              shuffle=True,
                              num_workers=2)

    model = Model(len(voices_loader.speakers))

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=consts.alpha, eps=1e-07)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def save_checkpoint(state, filename):
        torch.save(state, filename)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        for batch, speakers in train_loader:
            optimizer.zero_grad()
            loss = model(torch.tensor(batch), torch.tensor(speakers, dtype=torch.long))
            print(loss)
            loss.backward()
            optimizer.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename="checkpoint_e"+epoch+"pth.tar")


if __name__ == "__main__":
    main()
