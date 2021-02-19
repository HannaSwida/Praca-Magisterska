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
        embeddings = self.sinc_net(input) # TODO
        speakers_pred = self.classifier(embeddings)
        embeddings = embeddings.reshape((128, 3, constants.embedding_size))
        S1U1 = embeddings[:, 0, :]
        S1U2 = embeddings[:, 1, :]
        S2U1 = embeddings[:, 2, :]
        posp = torch.cat((S1U1, S1U2), 1)
        negp = torch.cat((S1U1, S2U1), 1)
        score_posp = self.discriminator(posp)
        score_negp = self.discriminator(negp)

        loss = (1 - score_posp) ** 2 + score_negp ** 2 + F.cross_entropy(speakers_pred, speakers.reshape((-1)))
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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sinc_conv = nn.Conv1d(1, 80, 251)# TODO: Sinc-based?
        self.conv2 = nn.Conv1d(80, 60, 5)
        self.conv3 = nn.Conv1d(60, 60, 5)
        self.lr = nn.LeakyReLU(0.1) #
        self.fc1 = nn.Linear(60, constants.embedding_size*2)
        self.fc2 = nn.Linear(constants.embedding_size*2, constants.embedding_size)

    def forward(self, x):
        x = self.sinc_conv(x) #TODO
        x = self.conv2(x)
        x = self.conv3(x)
        # TODO: x = layer_norm
        x = x.mean(2)
        x = self.fc1(x)
        x = self.lr(x)
        x = self.fc2(x)
        x = self.lr(x)
        # TODO: x = leaky_relu(x)
        return x


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
