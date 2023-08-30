import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from loader_classifier import VCset
from sklearn.preprocessing import LabelEncoder
import math
import torch.nn.functional as F
from model import Encoder
import os


class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(CustomMLP, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


def load_encoder(encoder_path):
    checkpoint = torch.load(encoder_path, map_location=torch.device('cpu'))
    encoder = Encoder()
    encoder.load_state_dict(checkpoint)
    return encoder


def extract_d_vectors(encoder, data_loader):
    d_vectors = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels_batch = batch
            c_1 = inputs.float()
            encoded_output1 = encoder(c_1)
            d_vectors.extend(encoded_output1.cpu().numpy())
            labels.extend(labels_batch)

    return np.array(d_vectors), np.array(labels)


def main(argv):
    n_batches = 3000
    batch_size = 128
    n_workers = 1
    n_epochs = 1000
    lr = 0.001
    learning_rate_decay = 0.95
    patience = 10000
    max_grad_norm = 1.0
    print("setting end")

    input_dirs_file = '../../source_vox1dir.txt'
    tr_data = VCset(input_dirs_file, n_batches=n_batches, chunk_size=0.4)
    print("tr data init end")
    tr_loader = DataLoader(tr_data, batch_size=batch_size, pin_memory=True, num_workers=n_workers)
    print("tr loader init end")
    encoder_path = "../checkpoints/MiModel/04s/latest/vox1_1000B_1500/small/encoder_108.pth"

    encoder = load_encoder(encoder_path)
    d_vectors, labels = extract_d_vectors(encoder, tr_loader)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    print("y_encoded end")

    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(d_vectors, y_encoded, test_size=0.2,
                                                                        random_state=42)

    input_dim = d_vectors.shape[1]
    output_dim = len(np.unique(labels))

    model = CustomMLP(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, nesterov=True)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}/{n_epochs}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Stop, żadnej poprawy od", patience, "epok")
            break

    outputdir = '../checkpoints/MiClassifier/'
    if outputdir != "":
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        path = os.path.join(outputdir, "sss" + str(epoch) + ".tar")

        model_state_dict = model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss,
            'nlabels': tr_data.get_nlabels()
        }, path)

    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted_labels = torch.max(outputs, 1)
        print(
            "*******************************************************************************************************************")
        print(
            "*******************************************************************************************************************")
        print("suma dobrze przewidzianych: ",
              torch.sum(predicted_labels == torch.tensor(y_test_encoded, dtype=torch.long)).item())
        print("Len: ", len(y_test_encoded))
        accuracy = torch.sum(predicted_labels == torch.tensor(y_test_encoded, dtype=torch.long)).item() / len(
            y_test_encoded)
        print("Dokładność:", accuracy)
