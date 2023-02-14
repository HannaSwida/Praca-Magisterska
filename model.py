import torch
import torch.nn as nn
from dnn_models import SincNet
import configparser

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, data_tensor):
        #Zaenkodować osobno c_1, c_2 i c_3 z data_tensora i jakoś wysłać do dekodera


        encoded_output = self.encoder(data_tensor)
        decoded_output = self.decoder(encoded_output)
        return decoded_output


class Encoder(nn.Module):

    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        self.sinc_layer = SincNet()
        self.conv1 = nn.Conv1d(80, 60, 5)
        self.conv2 = nn.Conv1d(60, 60, 5)
        self.ln1 = nn.LayerNorm(input_shape)
        self.ln2 = nn.LayerNorm(60)
        self.fc1 = nn.Linear(60 * input_shape[1], 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        print(type(x), "model x (powinien być Tensor)")
        x = self.sinc_layer(x)
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, output_dim)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        