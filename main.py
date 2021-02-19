from loader import Loader
from model import Model
import torch
import torch.optim as optim
import constants as consts

def main():
    loader = Loader("training-data/voxceleb")
    model = Model(len(loader.speakers))
    optimizer = optim.RMSprop(model.parameters(), lr=consts.learning_rate, alpha=consts.alpha, eps=1e-07)

    model.train()

    for epoch in range(100):
        batch, speakers = loader.next()
        optimizer.zero_grad()
        loss = model(torch.tensor(batch), torch.tensor(speakers, dtype=torch.long))
        print(loss)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
