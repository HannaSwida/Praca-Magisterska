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
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(CustomMLP, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def load_encoder(encoder_path):
    checkpoint = torch.load(encoder_path)
    encoder = Encoder()
    encoder.load_state_dict(checkpoint)
    return encoder


def extract_d_vectors(encoder, data_loader):
    d_vectors = []
    labels = []  # List to store the labels
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels_batch = batch
            c_1 = inputs.float()
            encoded_output1 = encoder(c_1)
            d_vectors.extend(encoded_output1.cpu().numpy())
            labels.extend(labels_batch)

    return np.array(d_vectors), np.array(labels)


def main(argv):
    n_batches = 10000
    batch_size = 128
    n_workers = 1
    # n_epochs = 200
    n_epochs = 100
    lr = 0.001
    learning_rate_decay = 0.95
    patience = 100
    max_grad_norm = 1.0
    print("n_batches: ", n_batches, "batch_size: ", batch_size, "n ep: ", n_epochs, "patience: ", patience)

    input_dirs_file = '../../source_vox1dir.txt'
    print(input_dirs_file)
    tr_data = VCset(input_dirs_file, n_batches=n_batches, chunk_size=0.4)
    tr_loader = DataLoader(tr_data, batch_size=batch_size, pin_memory=True, num_workers=n_workers)
    encoder_path = "../checkpoints/MiModel/04s/vox1_1000B_1500/small/encoder_141.pth"

    encoder = load_encoder(encoder_path)
    d_vectors, labels = extract_d_vectors(encoder, tr_loader)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(d_vectors, y_encoded, test_size=0.2,
                                                                        random_state=42)

    input_dim = d_vectors.shape[1]
    output_dim = len(np.unique(labels))
    hidden_dim = 8192
    print("Hidden dimensions size: ", hidden_dim)

    model = CustomMLP(input_dim, output_dim, hidden_dim)

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, nesterov=True) 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay) 
    print("optimizer: SGD") 
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)
    print("optimizer: Adam")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_test = torch.tensor(y_test_encoded, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train_encoded, dtype=torch.long)
    print("TESTED WITH test set")

    best_loss = float('inf')
    patience_counter = 0
    print("Training epochs start...")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        if epoch % 2 == 0:
            print(f"Epoch: {epoch}/{n_epochs}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping! No improvement in the last", patience, "epochs.")
            break

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                _, predicted_labels = torch.max(outputs, 1)
                accuracy = torch.sum(predicted_labels == torch.tensor(y_test, dtype=torch.long)).item() / len(y_test)
                good = 0
                label_mapping = {}
                with torch.no_grad():
                    outputs = model(X_test)
                    _, predicted_labels = torch.max(outputs, 1)

                    for i in range(len(y_test)):
                        true_label = y_test[i].item()
                        predicted_label = predicted_labels[i].item()
                        label_mapping[predicted_label] = true_label
                        if predicted_label == true_label:
                            good += 1

                print(f"Epoch {epoch + 1}:")
                print("Accuracy:", accuracy)
                print("Predicted Label to True Label Mapping:")
                print(label_mapping)
                print("Correct pairs: ", good, "/", len(label_mapping))

                outputdir = '../checkpoints/MiClassifier/vox1res'
                if outputdir != "":
                    if not os.path.exists(outputdir):
                        os.makedirs(outputdir)

                    path = os.path.join(outputdir, "classifier_10000b_vox1" + str(epoch) + ".tar")

                    model_state_dict = model.state_dict()

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': loss,
                        'nlabels': tr_data.get_nlabels()
                    }, path)

    outputdir = '../checkpoints/MiClassifier/vox1res'
    if outputdir != "":
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        path = os.path.join(outputdir, "vox1Classifier" + str(epoch) + ".tar")

        model_state_dict = model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss,
            'nlabels': tr_data.get_nlabels()
        }, path)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        print("OUTPUTS", outputs)
        _, predicted_labels = torch.max(outputs, 1)
        print("predicted labels: ", predicted_labels)
        print(
            "*****************************************************0.4s**************************************************************")
        print(
            "*******************************************************************************************************************")
        print(
            "*******************************************************************************************************************")
        print("predicted labels sum: ", torch.sum(predicted_labels == torch.tensor(y_test, dtype=torch.long)).item())
        print("Len: ", len(y_test))
        accuracy = torch.sum(predicted_labels == torch.tensor(y_test, dtype=torch.long)).item() / len(y_test)
        print("Accuracy:", accuracy)
        label_mapping = {}

        with torch.no_grad():
            outputs = model(X_test)
            print("OUTPUTS", outputs)
            _, predicted_labels = torch.max(outputs, 1)

            for i in range(len(y_test)):
                true_label = y_test[i].item()
                predicted_label = predicted_labels[i].item()

                label_mapping[predicted_label] = true_label

            print("Predicted Label to True Label Mapping:")
            print(label_mapping)


if __name__ == "__main__":
    main(None)