import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gensim
from gensim.models import Word2Vec
import torch.nn.functional as F

# Function to read file content line by line and append
def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines

# Function to preprocess the text
def preprocess(text):
    tokens = gensim.utils.simple_preprocess(text)
    return tokens

# Load the trained Word2Vec model
def load_word2vec_model(model_path):
    return Word2Vec.load(model_path)

# Prepare data loader for inference
def prepare_data_loader(sentence_list, word2vec_model):
    tokenized_sentences = [preprocess(sentence) for sentence in sentence_list]
    word_indices = [[word2vec_model.wv.key_to_index.get(word, 0) for word in sentence] for sentence in tokenized_sentences]
    max_length = max(len(sentence) for sentence in word_indices)
    padded_data = [sentence + [0] * (max_length - len(sentence)) for sentence in word_indices]
    return DataLoader(TensorDataset(torch.tensor(padded_data, dtype=torch.long)), batch_size=1, shuffle=False)

def print_predictions_table(classifier, sentences, predictions):
    print("\nPredictions:")
    print("-" * 112)
    print("| {:<13} | {:<80} | {:<10} |".format("Activation", "Sentence", "Prediction"))
    print("-" * 112)
    for sentence, prediction in zip(sentences, predictions):
        prediction_label = 'Positive' if prediction == 1 else 'Negative'
        print("| {:<13} | {:<80} | {:<10} |".format(classifier, sentence, prediction_label))
    print("-" * 112)

# Function to perform inference
def classify_sentences(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs[0])
            outputs = F.softmax(outputs, dim=1)  # Apply softmax
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist()) 
    return predictions
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, activation_fn, embeddings):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.activation = activation_fn
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  # Average embedding
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
# Main function for inference script
def main(sentence_file, model_type):
    model_dir = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a4/data"
    word2vec_model_path = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a3/data"

    model_path = os.path.join(model_dir, f'nn_{model_type}.model')
    word2vec_model = load_word2vec_model(os.path.join(word2vec_model_path, 'w2v.model'))
    sentences = read_file(sentence_file)

    # Prepare data loader for inference
    data_loader = prepare_data_loader(sentences, word2vec_model)

    # Load the entire model
    model = torch.load(model_path)

    # Classify sentences
    predictions = classify_sentences(model, data_loader)

    # Print predictions
    print_predictions_table(model_type, sentences, predictions)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 a4/inference.py a4/data/sentence.txt relu <path_to_sentence_file> <model_type>")
        sys.exit(1)

    sentence_file = sys.argv[1]
    model_type = sys.argv[2]
    main(sentence_file, model_type)
