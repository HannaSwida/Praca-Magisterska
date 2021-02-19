import torch.nn as nn
import torch as torch
import torch.nn.functional as F
import constants

class Model(nn.Module):
    def __init__(self, num_speakers):
        super(Model, self).__init__()

        self.discriminator = Discriminator()
        #self.sinc_net = SincNet() <--- TODO SincNET z Github
        self.classifier = Classifier(num_speakers)

    def forward(self, input, speakers):
        input = input.reshape((128 * 3, 1, 3200))
        embeddings = self.sinc_net(input)
        speakers_pred = self.classifier(embeddings)
        embeddings = embeddings.reshape((128, 3, constants.embedding_size))
        S1U1 = embeddings[:, 0, :]
        S1U2 = embeddings[:, 1, :]
        S2U1 = embeddings[:, 2, :]
        pp = torch.cat((S1U1, S1U2), 1)#pospara
        np = torch.cat((S1U1, S2U1), 1)#negpara
        score_pp = self.discriminator(pp)
        score_np = self.discriminator(np)

        loss = (1 - score_pp) ** 2 + score_np ** 2 + F.cross_entropy(speakers_pred, speakers.reshape((-1)))
        # TODO
        return loss.mean(0)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2 * constants.embedding_size, 2 * constants.embedding_size)
        self.fc2 = nn.Linear(2 * constants.embedding_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class Classifier(nn.Module):
    def __init__(self, num_speakers):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(constants.embedding_size, constants.embedding_size)
        self.fc2 = nn.Linear(constants.embedding_size, num_speakers)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        result = F.softmax(x, dim=1)
        return result
