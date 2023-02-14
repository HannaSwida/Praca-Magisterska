import torch.nn as nn
import torch as torch
import torch.nn.functional as Fun
from dnn_models import sinc_conv, SincNet
import constants
from constants import BATCHES


class Model(nn.Module):
    def __init__(self, num_speakers):
        super(Model, self).__init__()

        self.discriminator = Discriminator()
        self.encoder = Encoder()

    def generateDVec(self, x):
        batch_size, num_chunks, features  = x.shape
        x = x.reshape((batch_size * num_chunks, 1, features))
        embeddings = self.encoder(x)  # TODO
        embeddings = embeddings.reshape(batch_size, num_chunks, -1)
        return self.classifier.generateVector(embeddings) #dvectors

    def forward(self, input, speakers):
        input = input.reshape((BATCHES * 3, 1, 3200))
        embeddings = self.encoder(input)  # TODO
        #speakers_probs = self.classifier(embeddings)
        embeddings = embeddings.reshape((BATCHES, 3, constants.embedding_size))
        S1U1 = embeddings[:, 0, :]
        S1U2 = embeddings[:, 1, :]
        Srand = embeddings[:, 2, :]
        posp = torch.cat((S1U1, S1U2), 1)
        negp = torch.cat((S1U1, Srand), 1)
        score_posp = self.discriminator(posp)
        score_negp = self.discriminator(negp)
        return score_posp, score_negp, speakers


from data_io import read_conf_inp,str_to_bool
cfg_file= '../data_for_sinc.cfg'  # Config file of the speaker-id experiment used to generate the model

options=read_conf_inp(cfg_file)

fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))

CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,
          }
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.sinc_convolution = sinc_conv(80, 251, 16000)
        #self.convolution2 = nn.Conv1d(80, 60, 5)
        #self.convolution3 = nn.Conv1d(60, 60, 5)
        #self.lr = nn.LeakyReLU(0.1)
        #self.fc1 = nn.Linear(60, 2942)
        #self.fc2 = nn.Linear(constants.embedding_size * 2, constants.embedding_size)
        #
        #self.input_norm = nn.LayerNorm((1, 3200))
        #self.layer_norm1 = nn.LayerNorm((80, 2950))
        #self.layer_norm2 = nn.LayerNorm((60, 2946))
        #self.layer_norm3 = nn.LayerNorm((60, 2942))

        self.sinc_convolution = sinc_conv(80, 251, 16000)
        self.convolution2 = nn.Conv1d(80, 60, 5)
        self.convolution3 = nn.Conv1d(60, 60, 5)
        self.lr = nn.LeakyReLU(2048)
        self.lr = nn.LeakyReLU(1024)
        self.normalization1 = nn.LayerNorm([384, 1, 3200])
        self.normalization2 = nn.LayerNorm([384, 80, 2950])
        self.normalization3 = nn.LayerNorm([384, 60, 2942])
        self.normalization4 = nn.LayerNorm([384, 60, 2946])
        self.normalization5 = nn.LayerNorm([384, 60, 2946])
       #self.batchNorm1 = nn.BatchNorm2d()(x)
       #self.batchNorm1 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.normalization1(x) #(nie jestem pewna czy normalizacja input tutaj czy po sincNet)
        x =SincNet(CNN_arch)
        x = self.sinc_convolution(x)
        x = self.normalization2(x)
        x = self.convolution2(x)
        x = self.normalization4(x)
        x = self.convolution3(x)
        x = self.normalization3(x)
        x = self.lr(x)
        x = self.normalization4(x)
        x = self.lr(x)
        x = self.batchNorm2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2 * constants.embedding_size, 2 * constants.embedding_size)
        self.proj = nn.Linear(2 * constants.embedding_size, 1)

    def forward(self, x):
        x = Fun.relu(self.fc1(x))
        x = torch.sigmoid(x)
        return x
