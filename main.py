from voxceleb_loader import VoxCelebLoader
from model import Model
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import constants as consts
import argparse
import os
from constants import BATCHES_PER_EPOCH
import configparser


def make_batch(items):
    samples = [item[0] for item in items]
    speakers = [item[1] for item in items]

    return torch.Tensor(samples), torch.Tensor(speakers)

parser = argparse.ArgumentParser(description='Praca magisterska')
parser.add_argument('--loader', default='voxceleb',
                    help='dataset type')
parser.add_argument('--data', default='vox1/test',
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
args = parser.parse_args()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#TODO: Do sprawdzenia
    input_dim = (80, 251)
    hidden_dim = 1024
    output_dim = 40  # This can be changed based on the number of classes in the target speaker labels
    configpars = configparser.ConfigParser()
    configpars.read('SincNet.cfg')
    config = configpars
    print(config['cnn']['cnn_N_filt'], "fsfsfs")
    # Initialize the model
    model = Model(input_dim, hidden_dim, output_dim)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, eps=1e-07)

    criterion = torch.nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    model.train()
    train_loss = 0
    for i, (data_tensor, target) in enumerate(train_loader):
        
        print("data: ", type(data_tensor), "target: ", type(target), "main.py:83")
        optimizer.zero_grad()
        output = model(data_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()