import os
import sys
import random

# List of Stop Words
# list collected from 
# from nltk.corpus import stopwords
# ", ".join(stopwords.words('english'))

stop_words = [
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
]

# Function to tokenize the corpus
def tokenize_corpus(text):
    tokens = text.split()
    return tokens
    
#Remove the special characters
def remove_special_characters(tokens):
    special_characters = set("!#$%&()*+/:,;.<=>@[\\]^`{|}~\t\n")
    remove_sc = [''.join(char for char in token if char not in special_characters) for token in tokens]
    return remove_sc

# Function to remove stop words
def remove_stop_words(tokens):
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Function to read file content line by line and append
def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip()) 
    return lines



def split_data(tokenizer2, datalabel):
    tokenizer = tokenizer2.copy()
    data_with_labels = list(zip(tokenizer, datalabel))
    random.shuffle(data_with_labels)
    
    # Calculate the sizes of each split
    total_size = len(data_with_labels)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # Split the data and labels
    train_set = data_with_labels[:train_size]
    val_set = data_with_labels[train_size:train_size + val_size]
    test_set = data_with_labels[train_size + val_size:]

    # Unzip the data and labels
    train_data, train_labels = zip(*train_set)
    val_data, val_labels = zip(*val_set)
    test_data, test_labels = zip(*test_set)

    return list(train_data), list(train_labels), list(val_data), list(val_labels), list(test_data), list(test_labels)
  

# Function to write tokenized sentences to a CSV file
def write_to_csv(directory, filename, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        for tokens in data:
            sentence = ','.join(tokens).strip()+'.\n'
            csvfile.write(sentence)

# Function to write labels sentences to a CSV file
def write_to_csvlabel(directory, filename, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        for label in data:
            labels = label + '\n'
            csvfile.write(labels)

def main(data_dir):

    # Read text of the files separately
    pos_text = read_file(os.path.join(data_dir, 'pos.txt'))
    neg_text = read_file(os.path.join(data_dir, 'neg.txt'))

    # Initialize empty lists for processed text
    pos_processed = []
    neg_processed = []
    # Tokenize and remove special characters from each line
    pos_processed = [remove_special_characters(tokenize_corpus(line)) for line in pos_text]
    neg_processed = [remove_special_characters(tokenize_corpus(line)) for line in neg_text]
      
    # Create labels
    pos_labels = ['pos'] * len(pos_processed)
    neg_labels = ['neg'] * len(neg_processed)
    
    #Concatenates the pos and neg dataset 
    concat = pos_processed + neg_processed
    labels = pos_labels + neg_labels


    # Split data into train (80%), val(10%), test(10%)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(concat, labels)

    # Write files (.csv files) w stopwords
    write_to_csv(data_dir, 'out.csv', concat)
    write_to_csv(data_dir, 'train.csv', train_data)
    write_to_csv(data_dir, 'val.csv', val_data)
    write_to_csv(data_dir, 'test.csv', test_data)
    
    # Remove stop words
    concat_stop_word = [remove_stop_words(line) for line in concat]
    train_stop_word = [remove_stop_words(line) for line in train_data]
    val_stop_word = [remove_stop_words(line) for line in val_data]
    test_stop_word = [remove_stop_words(line) for line in test_data]

    # Write files (.csv files) w/o stopwords
    write_to_csv(data_dir, 'out_ns.csv', concat_stop_word)
    write_to_csv(data_dir, 'train_ns.csv', train_stop_word)
    write_to_csv(data_dir, 'val_ns.csv', val_stop_word)
    write_to_csv(data_dir, 'test_ns.csv', test_stop_word)

    #write the (.csv files) for labels 
    write_to_csvlabel(data_dir, 'train_labels.csv', train_labels)
    write_to_csvlabel(data_dir, 'val_labels.csv', val_labels)
    write_to_csvlabel(data_dir, 'test_labels.csv', test_labels)





if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 a1/main.py a1/data <Path to data_dir>") 
        sys.exit(1)

    data_dir = sys.argv[1]
    main(data_dir)