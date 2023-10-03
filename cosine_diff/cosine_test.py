import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from loader_test import VCset
from sklearn.preprocessing import LabelEncoder
import math
import pandas as pd
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from models import Classifier, Encoder, SincConv
import os
import neat_helper
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


def load_encoder(encoder_path):
    checkpoint = torch.load(encoder_path, map_location=torch.device('cpu'))
    encoder = Encoder()
    encoder.load_state_dict(checkpoint)
    return encoder


def load_classifier(classifier_path, input_dim, output_dim, hidden_dim):
    checkpoint = torch.load(classifier_path, map_location=torch.device('cpu'))

    classifier = Classifier(input_dim, output_dim, hidden_dim).to("cpu")

    classifier.load_state_dict(checkpoint['model_state_dict'])

    return classifier


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


def extract_d_vectors_classifier(classifier, tensor, labels):
    tensor = tensor.float()
    with torch.no_grad():
        encoded_classifier_output = classifier(tensor)
        d_vectors = encoded_classifier_output.cpu().numpy()

    return np.array(d_vectors), np.array(labels)


def print_real_labels(labels):
    print("Real Labels:", labels)


def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = 1.0
    for i in range(len(fpr)):
        if fpr[i] >= 1.0 - tpr[i] and eer >= 1.0 - tpr[i]:
            eer = 1.0 - tpr[i]
    return eer



def save_similarity_matrix(similarity_matrix, labels, file_path):
    similarity_matrix_first_30 = similarity_matrix[:30, :30]
    labels_first_30 = labels[:30]

    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix_first_30, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(len(labels_first_30)), labels_first_30, rotation=90)
    plt.yticks(range(len(labels_first_30)), labels_first_30)
    plt.title('Cosine Similarity Matrix')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def main(argv):
    encoder = load_encoder(neat_helper.encoder_path)
    tr_data = VCset(neat_helper.input_dirs_file, chunk_size=0.4)

    tr_loader = DataLoader(tr_data, batch_size=len(tr_data), pin_memory=True, num_workers=neat_helper.n_workers)

    d_vectors_encoder, labels_enc = extract_d_vectors(encoder, tr_loader)

    input_dim = d_vectors_encoder.shape[1]
    output_dim = len(np.unique(labels_enc))
    hidden_dim = 2048
    print(input_dim, output_dim)
    classifier = Classifier(input_dim, output_dim, hidden_dim)
    classifier.load_state_dict(neat_helper.checkpoint['model_state_dict'])

    classifier.eval()

    d_vectors_tensor = torch.Tensor(d_vectors_encoder)
    d_vectors_classifier, labels_classifier = extract_d_vectors_classifier(classifier, d_vectors_tensor, labels_enc)

    print("Shape of d_vectors_encoder:", d_vectors_encoder.shape)
    print("Shape of d_vectors_classifier:", d_vectors_classifier.shape)

    similarity_scores = cosine_similarity(d_vectors_encoder, d_vectors_classifier)

    labels_similarity = np.where(labels_enc[:, None] == labels_classifier[None, :], 1, 0)

    similarity_scores = similarity_scores.flatten()
    labels_similarity = labels_similarity.flatten()

    eer = calculate_eer(labels_similarity, similarity_scores)
    print("Equal Error Rate (EER):", eer)


if __name__ == "__main__":
    main(None)
