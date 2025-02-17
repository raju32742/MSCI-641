import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def load_data(data_dir, filename):
    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f]
    return data

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, vectorizer, filename):
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    best_alpha = None
    best_accuracy = 0
    best_model = None
    
    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_pred)        
        if accuracy > best_accuracy:
            best_alpha = alpha
            best_accuracy = accuracy
            best_model = model
    test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # Save the best model and vectorizer's information to a file
    model_data = {
        'model': best_model,
        'vocabulary': vectorizer.vocabulary_,
        'ngram_range': vectorizer.ngram_range,
        'token_pattern': vectorizer.token_pattern
    }

    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)
    
    return best_accuracy, test_accuracy

def print_results_in_table(test_acc_uni, test_acc_bi, test_acc_uni_bi, test_acc_uni_ns, test_acc_bi_ns, test_acc_uni_bi_ns):
    table = [
        ["Stopwords removed", "Text features", "Accuracy (test set)"],
        ["yes", "unigrams", f"{test_acc_uni_ns:.6f}"],
        ["yes", "bigrams", f"{test_acc_bi_ns:.6f}"],
        ["yes", "unigrams+bigrams", f"{test_acc_uni_bi_ns:.6f}"],
        ["no", "unigrams", f"{test_acc_uni:.6f}"],
        ["no", "bigrams", f"{test_acc_bi:.6f}"],
        ["no", "unigrams+bigrams", f"{test_acc_uni_bi:.6f}"],
    ]
    
    header = table[0]
    rows = table[1:]
    
    # Print the table with separators
    print("-" * 62)
    print("| {:<18} | {:<18} | {:<18} |".format(*header))
    print("-" * 61)
    for row in rows:
        print("| {:<18} | {:<18} | {:<18} |".format(*row))
    print("-" * 62)

def print_accuracy_table(file_names, val_accuracies, test_accuracies):
    print("-" * 60)
    print("| {:<18} | {:<18} | {:<18} |".format("File name", "Validation Acc", "Test Acc"))
    print("-" * 60)
    for file_name, val_acc, test_acc in zip(file_names, val_accuracies, test_accuracies):
        print("| {:<18} | {:<18.6f} | {:<18.6f} |".format(file_name, val_acc, test_acc))
    print("-" * 60)

def main(data_dir, out_dir):
    train_data = load_data(data_dir, 'train.csv')
    val_data = load_data(data_dir, 'val.csv')
    test_data = load_data(data_dir, 'test.csv')

    train_data_ns = load_data(data_dir, 'train_ns.csv')
    val_data_ns = load_data(data_dir, 'val_ns.csv')
    test_data_ns = load_data(data_dir, 'test_ns.csv')

    train_labels = load_data(data_dir, 'train_labels.csv')
    val_labels = load_data(data_dir, 'val_labels.csv')
    test_labels = load_data(data_dir, 'test_labels.csv')

    # Vectorizers for data with stopwords
    uni_vectorizer_sw = CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    bi_vectorizer_sw = CountVectorizer(ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b")
    uni_bi_vectorizer_sw = CountVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")

    # Vectorizers for data without stopwords
    uni_vectorizer_ns = CountVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    bi_vectorizer_ns = CountVectorizer(ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b")
    uni_bi_vectorizer_ns = CountVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")

    # Fit and transform data with stopwords
    train_uni = uni_vectorizer_sw.fit_transform(train_data)
    val_uni = uni_vectorizer_sw.transform(val_data)
    test_uni = uni_vectorizer_sw.transform(test_data)

    train_bi = bi_vectorizer_sw.fit_transform(train_data)
    val_bi = bi_vectorizer_sw.transform(val_data)
    test_bi = bi_vectorizer_sw.transform(test_data)

    train_uni_bi = uni_bi_vectorizer_sw.fit_transform(train_data)
    val_uni_bi = uni_bi_vectorizer_sw.transform(val_data)
    test_uni_bi = uni_bi_vectorizer_sw.transform(test_data)

    # Fit and transform data without stopwords
    train_ns_uni = uni_vectorizer_ns.fit_transform(train_data_ns)
    val_ns_uni = uni_vectorizer_ns.transform(val_data_ns)
    test_ns_uni = uni_vectorizer_ns.transform(test_data_ns)

    train_ns_bi = bi_vectorizer_ns.fit_transform(train_data_ns)
    val_ns_bi = bi_vectorizer_ns.transform(val_data_ns)
    test_ns_bi = bi_vectorizer_ns.transform(test_data_ns)

    train_ns_uni_bi = uni_bi_vectorizer_ns.fit_transform(train_data_ns)
    val_ns_uni_bi = uni_bi_vectorizer_ns.transform(val_data_ns)
    test_ns_uni_bi = uni_bi_vectorizer_ns.transform(test_data_ns)
# Training MNB with w stopwords..."
    val_acc_uni, test_acc_uni = train_and_evaluate_model(train_uni, train_labels, val_uni, val_labels, test_uni, test_labels, uni_vectorizer_sw, os.path.join(out_dir, 'mnb_uni.pkl'))
    val_acc_bi, test_acc_bi = train_and_evaluate_model(train_bi, train_labels, val_bi, val_labels, test_bi, test_labels, bi_vectorizer_sw, os.path.join(out_dir, 'mnb_bi.pkl'))
    val_acc_uni_bi, test_acc_uni_bi = train_and_evaluate_model(train_uni_bi, train_labels, val_uni_bi, val_labels, test_uni_bi, test_labels, uni_bi_vectorizer_sw, os.path.join(out_dir, 'mnb_uni_bi.pkl'))

#Training MNB with w/o stopwords..."
    val_acc_uni_ns, test_acc_uni_ns = train_and_evaluate_model(train_ns_uni, train_labels, val_ns_uni, val_labels, test_ns_uni, test_labels, uni_vectorizer_ns, os.path.join(out_dir, 'mnb_uni_ns.pkl'))
    val_acc_bi_ns, test_acc_bi_ns = train_and_evaluate_model(train_ns_bi, train_labels, val_ns_bi, val_labels, test_ns_bi, test_labels, bi_vectorizer_ns, os.path.join(out_dir, 'mnb_bi_ns.pkl'))
    val_acc_uni_bi_ns, test_acc_uni_bi_ns = train_and_evaluate_model(train_ns_uni_bi, train_labels, val_ns_uni_bi, val_labels, test_ns_uni_bi, test_labels, uni_bi_vectorizer_ns, os.path.join(out_dir, 'mnb_uni_bi_ns.pkl'))
    
    file_names = ['mnb_uni.pkl', 'mnb_bi.pkl', 'mnb_uni_bi.pkl', 'mnb_uni_ns.pkl', 'mnb_bi_ns.pkl', 'mnb_uni_bi_ns.pkl']
    val_accuracies = [val_acc_uni, val_acc_bi, val_acc_uni_bi, val_acc_uni_ns, val_acc_bi_ns, val_acc_uni_bi_ns]
    test_accuracies = [test_acc_uni, test_acc_bi, test_acc_uni_bi, test_acc_uni_ns, test_acc_bi_ns, test_acc_uni_bi_ns]

    print_accuracy_table(file_names, val_accuracies, test_accuracies)
    print_results_in_table(test_acc_uni, test_acc_bi, test_acc_uni_bi, test_acc_uni_ns, test_acc_bi_ns, test_acc_uni_bi_ns)

if __name__ == "__main__":
    if len(sys.argv) != 3:
    print("Usage: python3 a2/main.py a1/data a2/data <Path to output_dir>") 
        sys.exit(1)

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    main(data_dir, out_dir)
