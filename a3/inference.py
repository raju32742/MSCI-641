import os
import sys
from gensim.models import Word2Vec

def main(words_file):
    model_path = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a3/data/w2v.model"
    model = Word2Vec.load(model_path)
    
    with open(words_file, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]

    for word in words:
        if word in model.wv.key_to_index:
            similar_words = model.wv.most_similar(word, topn=20)
            print(f"|{'-' * 32}|")
            print(f"| Words most similar to '{word}'    |")
            print(f"|{'-' * 32}|")
            print(f"| {'Word':<18} | {'Score':<9} |")
            print(f"|{'-' * 32}|")
            for similar_word, score in similar_words:
                print(f"| {similar_word:<18} | {score:<9.3f} |")
            print(f"|{'-' * 32}|\n")
        else:
            print(f"'{word}' not in vocabulary.\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 a3/inference.py a3/data/word.txt <Path to model> <Path to words_file>")
        sys.exit(1)
    words_file = sys.argv[1]
    main(words_file)

