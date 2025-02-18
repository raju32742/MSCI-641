import os
import sys
import random
import gensim
from gensim.models import Word2Vec

# Function to read file content line by line and append
def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines

# Function to tokenize the corpus
def preprocess(text):
    tokens = gensim.utils.simple_preprocess(text)
    return tokens

def main(data_dir):
    out_dir = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a3/data"
    pos_text = read_file(os.path.join(data_dir, 'pos.txt'))
    neg_text = read_file(os.path.join(data_dir, 'neg.txt'))

    corpus = pos_text + neg_text

    # Preprocess the text data
    preprocessed_text = [preprocess(line) for line in corpus]

    # Train Word2Vec model
    model = Word2Vec(sentences=preprocessed_text, vector_size=100, window=5, min_count=2, workers=4)
    model.save(os.path.join(out_dir, 'w2v.model'))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 a3/main.py a1/data <Path to data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]
    main(data_dir)