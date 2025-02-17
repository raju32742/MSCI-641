import sys
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

    # Define stop words
stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd",
        "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
        "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've",
        "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
        "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
        "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
        "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
    ])

def load_sentences(filename):
    """Load sentences from a file, stripping whitespace and newlines."""
    with open(filename, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    return sentences

def load_model(model_path):
    """Load a pickled model and its associated data from the specified path."""
    if not os.path.isfile(model_path):
        print(f"Model file '{model_path}' does not exist.")
        sys.exit(1)
    
    with open(model_path, 'rb') as model_file:
        model_data = pickle.load(model_file)
    
    return model_data

def remove_special_characters(tokens):
    special_characters = set("!#$%&()*+/:,;.<=>@[\\]^`{|}~\t\n")
    return [''.join(char for char in token if char not in special_characters) for token in tokens]

def remove_stop_words(tokens, stop_words):
    return [word for word in tokens if word.lower() not in stop_words]

def preprocess_text(text, stop_words=None):
    tokens = text.split()
    tokens = remove_special_characters(tokens)
    if stop_words is not None:
        tokens = remove_stop_words(tokens, stop_words)
    return ' '.join(tokens)

def print_predictions_table(classifier, sentences, predictions):
    print("\nPredictions:")
    print("-" * 112)
    print("| {:<13} | {:<80} | {:<10} |".format("Classifier", "Sentence", "Prediction"))
    print("-" * 112)
    for sentence, prediction in zip(sentences, predictions):
        prediction_label = 'Positive' if prediction == 'pos' else 'Negative'
        print("| {:<10} | {:<80} | {:<10} |".format(classifier, sentence, prediction_label))
    print("-" * 112)


def main(input_file, classifier_type):
    
    model_base_path = "/Users/raju/Raju Mac/UW/UW/Spring 24/MSCI 641/Assignment/a2/data/"
    model_path = os.path.join(model_base_path, f"{classifier_type}.pkl")
    
    # Load sentences from the input file
    sentences = load_sentences(input_file)
    
    # Load the specified model and its data
    model_data = load_model(model_path)
    model = model_data['model']
    vocabulary = model_data['vocabulary']
    ngram_range = model_data['ngram_range']
    token_pattern = model_data['token_pattern']
    
    # Determine if stopwords should be removed based on the classifier type
    remove_stopwords = classifier_type.endswith('_ns')

    # Preprocess sentences
    processed_sentences = [preprocess_text(sentence, stop_words if remove_stopwords else None) for sentence in sentences]

    # Recreate the CountVectorizer with the saved vocabulary
    vectorizer = CountVectorizer(vocabulary=vocabulary, 
                                 ngram_range=ngram_range, 
                                 token_pattern=token_pattern)
    
    # Transform sentences to feature vectors using the recreated vectorizer
    X = vectorizer.transform(processed_sentences)
    
    # Predict using the loaded model
    predictions = model.predict(X)
    
    # Print sentences and their predictions
    print_predictions_table(classifier_type, sentences, predictions)

if __name__ == "__main__":
    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python3 a2/inference.py a2/data/sentence.txt mnb_uni_bi")
        sys.exit(1)

    input_file = sys.argv[1]
    classifier_type = sys.argv[2]
    main(input_file, classifier_type)

