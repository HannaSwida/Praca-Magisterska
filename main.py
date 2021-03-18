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

    checkpoint = {
        "epoch": 90,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }

   # torch.save(checkpoint,"checkpoint.pth")

    loaded_chk = torch.load("checkpoint.pth")
    epoch = loaded_chk["epoch"]
    model = Model(len(vox_loader.speakers))
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])

    for epoch in range(100):
        for batch, speakers in vox_loader:
            optimizer.zero_grad()
            loss = model(torch.tensor(batch), torch.tensor(speakers, dtype=torch.long))
            print(loss)
            loss.backward()
            optimizer.step()

    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)

    loaded_model = Model(len(vox_loader.speakers))
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()


if __name__ == "__main__":
    main()
