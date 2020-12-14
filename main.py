from reader import Reader
from model import Model
import torch
import torch.optim as optim
import constants

def main():
    reader = Reader("training-data/voxceleb")
    model = Model(len(reader.speakers))
    optimizer = optim.SGD(model.parameters(), lr=constants.learning_rate,
                          momentum=constants.momentum)

    model.train()

    for epoch in range(100):
        audio, speakers = reader.next()
        optimizer.zero_grad()
        loss = model(torch.tensor(audio), torch.tensor(speakers, dtype=torch.long))
        print(loss)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
