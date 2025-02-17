# MSCI 641 Assignment 1


## Objective:
1)Tokenizing the corpus (https://github.com/fuzhenxin/textstyletransferdata/tree/master/sentiment).

2)Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n

3)Creating two versions of your dataset: (a) with stopwordsand (b) without stopwords.

4)Randomly spliting the data into training (80%), validation (10%) and test (10%) 


## Running the code:
To run the file, use the following command:
```python main.py```

This assumes that you have "pos.txt" and "neg.txt" files in the same location as the python file. The files are from https://github.com/fuzhenxin/textstyletransferdata/tree/master/sentiment.

If you have the above files at any other location, let's say they are at path1, and path2 (for pos.txt and neg.txt respectively), use:
```python main.py "path1" "path2"```

If you have a single parent directory where both "pos.txt" and "neg.txt" exist, then use the following command:
```python main.py "parent path"```

It is important to specify the paths in quotes.


In total there are 12 output files generated:

1) **out.csv**: Tokens for all the reviews without stop words.
2) **out_stop.csv**: Tokens for all the reviews with stop words.
3) **labels_all.csv**: Labels for all the reviews.
4) **train.csv**: Tokens for all the reviews in the training set without stop words.
5) **train_stop.csv**: Tokens for all the reviews in the training set with stop words.
6) **labels_train.csv**: Labels for all the reviews in the training set.
7) **val.csv**: Tokens for all the reviews in the validation set without stop words.
8) **val_stop.csv**: Tokens for all the reviews in the validation set with stop words.
9) **labels_val.csv**: Labels for all the in the validation set reviews.
10) **test.csv**: Tokens for all the reviews in the test set without stop words.
11) **test_stop.csv**: Tokens for all the reviews in the test set without stop words.
12) **labels_test.csv**: Labels for all the in the test set reviews.

In total there are 8,00,000 reviews which are split 8:1:1 into training, validation and test sets respectively.
