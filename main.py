from vox_celeb_loader import VoxCelebLoader
from model import Model
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import constants as consts


def main():
    vox_loader = VoxCelebLoader("training-data/voxceleb")
    train_loader = DataLoader(dataset=vox_loader,
                              shuffle=True,
                              num_workers=2)
    model = Model(len(vox_loader.speakers))
    optimizer = optim.RMSprop(model.parameters(), lr=consts.learning_rate, alpha=consts.alpha, eps=1e-07)

    model.train()

    for epoch in range(100):
        for batch, speakers in vox_loader:
            optimizer.zero_grad()
            loss = model(torch.tensor(batch), torch.tensor(speakers, dtype=torch.long))
            print(loss)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
