import torch.nn as nn
import torch as torch
import torch.nn.functional as F
import constants

class Model(nn.Module):
    def __init__(self, num_speakers):
        super(Model, self).__init__()

        self.discriminator = Discriminator()
        self.sinc_net = SincNet()
        self.classifier = Classifier(num_speakers)

    def forward(self, input, speakers):
        input = input.reshape((128 * 3, 1, 3200))
        embeddings = self.sinc_net(input)
        speakers_pred = self.classifier(embeddings)
        embeddings = embeddings.reshape((128, 3, constants.embedding_size))
        S1U1 = embeddings[:, 0, :]
        S1U2 = embeddings[:, 1, :]
        S2U1 = embeddings[:, 2, :]
        pp = torch.cat((S1U1, S1U2), 1)
        np = torch.cat((S1U1, S2U1), 1)
        score_pp = self.discriminator(pp)
        score_np = self.discriminator(np)

        loss = (1 - score_pp) ** 2 + score_np ** 2 + F.cross_entropy(speakers_pred, speakers.reshape((-1)))

        return loss.mean(0)

class SincNet(nn.Module):
    def __init__(self):
        super(SincNet, self).__init__()
        self.sinc_conv = nn.Conv1d(1, 80, 251)
        self.conv2 = nn.Conv1d(80, 60, 5)
        self.conv3 = nn.Conv1d(60, 60, 5)

        self.fc1 = nn.Linear(60, 2048)
        self.fc2 = nn.Linear(2048, constants.embedding_size)
    def forward(self, x):
        x = self.sinc_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # TODO: x = layer_norm
        x = x.mean(2)
        x = self.fc1(x)
        # TODO: x = leaky_relu(x)
        x = self.fc2(x)
        # TODO: x = leaky_relu(x)
        return x


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
        return F.softmax(x, dim=1)
