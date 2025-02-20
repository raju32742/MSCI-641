import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

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

# Prepare data loaders
def prepare_data_loaders(data_dir, word2vec_model, batch_size=1024):
    def load_data(file_path, labels_path):
        data = read_file(file_path)
        labels = read_file(labels_path)
        labels = LabelEncoder().fit_transform(labels)

        tokenized_data = [preprocess(line) for line in data]
        word_indices = [[word2vec_model.wv.key_to_index.get(word, 0) for word in line] for line in tokenized_data]
        max_length = max(len(line) for line in word_indices)

        padded_data = [line + [0] * (max_length - len(line)) for line in word_indices]
        return np.array(padded_data), np.array(labels)

    train_data, train_labels = load_data(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'train_labels.csv'))
    val_data, val_labels = load_data(os.path.join(data_dir, 'val.csv'), os.path.join(data_dir, 'val_labels.csv'))
    test_data, test_labels = load_data(os.path.join(data_dir, 'test.csv'), os.path.join(data_dir, 'test_labels.csv'))

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.long), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.long), torch.tensor(val_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.long), torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Neural network model
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, activation_fn, dropout_rate, embeddings):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


    return model

# Evaluate the model on the test set
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct_test / total_test
    return test_loss, test_accuracy

# Main function
def main(data_dir):
    out_dir = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a4/data"
    word2vec_model_path = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a3/data"
    # Load the trained Word2Vec model
    word2vec_model = load_word2vec_model(os.path.join(word2vec_model_path, 'w2v.model'))

    train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, word2vec_model)

    input_size = word2vec_model.vector_size
    hidden_size = 300
    output_size = 2
    dropout_rates = [0.3]
    num_epochs = 50
    learning_rate = 0.001
    l2_lambdas = [0]

    activations = {
        'relu': nn.ReLU()
        # 'sigmoid': nn.Sigmoid(),
        # 'tanh': nn.Tanh()
    }

    results = {}

    for act_name, activation in activations.items():
        best_val_accuracy = 0
        best_model_path = ""
        for dropout_rate in dropout_rates:
            for l2_lambda in l2_lambdas:
                print(f'Training with {act_name} activation, dropout rate: {dropout_rate}, L2 lambda: {l2_lambda}')
                model = SimpleNN(len(word2vec_model.wv), input_size, hidden_size, output_size, activation, dropout_rate, torch.tensor(word2vec_model.wv.vectors))
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

                # Train the model
                model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

                # Evaluate the model on the validation set
                val_loss, val_accuracy = evaluate(model, val_loader, criterion)
                # If this model has the best validation accuracy so far, save it
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_path = os.path.join(out_dir, f'nn_{act_name}.model')
                    torch.save(model, best_model_path)  # Save the entire model

                # Evaluate the model on the test set
                test_loss, test_accuracy = evaluate(model, test_loader, criterion)
                print(f'{act_name}, Dropout: {dropout_rate}, L2: {l2_lambda}, Test Accuracy: {test_accuracy:.4f}')

        results[act_name] = best_model_path

    print("Training completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 a4/main.py a1/data <Path to output_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]
    main(data_dir)
