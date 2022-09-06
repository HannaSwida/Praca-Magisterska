import torch.nn as nn
import torch as torch
import torch.nn.functional as Fun
from dnn_models import sinc_conv
import constants
from constants import BATCHES


class Model(nn.Module):
    def __init__(self, num_speakers):
        super(Model, self).__init__()

        self.discriminator = Discriminator()
        self.encoder = Encoder()
        self.classifier = Classifier(num_speakers)

    def generateDVec(self, x):
        batch_size, num_chunks, features  = x.shape
        x = x.reshape((batch_size * num_chunks, 1, features))
        embeddings = self.encoder(x)  # TODO
        embeddings = embeddings.reshape(batch_size, num_chunks, -1)
        return self.classifier.generateVector(embeddings) #dvectors

    def forward(self, input, speakers):
        print("input size", input.shape)
        input = input.reshape((BATCHES * 3, 1, 3200))
        embeddings = self.encoder(input)  # TODO
        speakers_probs = self.classifier(embeddings)
        embeddings = embeddings.reshape((BATCHES, 3, constants.embedding_size))
        S1U1 = embeddings[:, 0, :]
        S1U2 = embeddings[:, 1, :]
        Srand = embeddings[:, 2, :]
        posp = torch.cat((S1U1, S1U2), 1)
        negp = torch.cat((S1U1, Srand), 1)
        score_posp = self.discriminator(posp)
        score_negp = self.discriminator(negp)
        return score_posp, score_negp, speakers_probs, speakers


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sinc_convolution = sinc_conv(80, 251, 16000)
        self.convolution2 = nn.Conv1d(80, 60, 5)
        self.convolution3 = nn.Conv1d(60, 60, 5)
        self.lr = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(60, constants.embedding_size * 2)
        self.fc2 = nn.Linear(constants.embedding_size * 2, constants.embedding_size)

        self.input_norm = nn.LayerNorm((1, 3200))
        self.layer_norm1 = nn.LayerNorm((80, 2950))
        self.layer_norm2 = nn.LayerNorm((60, 2946))
        self.layer_norm3 = nn.LayerNorm((60, 2942))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.sinc_convolution(x)
        x = self.layer_norm1(x)
        x = self.convolution2(x)
        x = self.layer_norm2(x)
        x = self.convolution3(x)
        x = self.layer_norm3(x)
        x = x.mean(2)
        x = self.fc1(x)
        x = self.lr(x)
        x = self.fc2(x)
        out = self.lr(x)
        #print("out from Encoder forward {}".format(x))
        return out


class Classifier(nn.Module):
    def __init__(self, num_speakers):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(constants.embedding_size, constants.embedding_size) #inner layer
        self.proj = nn.Linear(constants.embedding_size, num_speakers)

    def generateVector(self, x):
        x = self.encode(x)
        x = torch.nn.functional.normalize(x, dim=-1)  # x: B ns, nF
        x = torch.mean(x, 1)
        return x

    def encode(self, x):
        return Fun.relu(self.fc1(x))

    def forward(self, x):
        x = self.encode(x) #hidden representation
        x = Fun.softmax(self.proj(x), dim=1)
       #print("x from Classifier forward {}".format(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2 * constants.embedding_size, 2 * constants.embedding_size)
        self.proj = nn.Linear(2 * constants.embedding_size, 1)

    def forward(self, x):
        #print("disc forward", x)
        x = Fun.relu(self.fc1(x))
       # print("disc forward relu", x)
        x = self.proj(x)
        x = torch.sigmoid(x)
        # print("disc forward proj", x)
        #print("x from Discriminator forward {}".format(x))
        return x
