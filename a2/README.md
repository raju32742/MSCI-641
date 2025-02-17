## Add execution instructions here
## Add execution instructions here
## To activate the Environment first write the command 
-------------------------------------------------------------------------------------
`source /home/gsahu/miniconda3/bin/activate /home/gsahu/miniconda3/envs/msci641_env/`

## To run the main.py file 
`python3 a2/main.py  /DATA1/smturaju/assignments/a1/data` 

## To Run the inference file 
## Classifier type could be :[ mnb_uni, mnb_bi, mnb_uni_bi, mnb_uni_ns, mnb_bi_ns mnb_uni_bi_ns ] 
`python3 a2/inference.py /DATA1/smturaju/assignments/a2/data/sentence.txt mnb_uni_bi`

## Train the MultinomialNB classifier and fine-tune with validation dataset to find the best model. Finally, calculate the 
## Test accuracy using test dataset using best model for each text feature format. 
```
|--------------------------------------------------------------|
| File name          | Validation Acc     | Test Acc           |
|--------------------------------------------------------------|
| mnb_uni.pkl        | 0.809100           | 0.807388           |
| mnb_bi.pkl         | 0.825625           | 0.824887           |
| mnb_uni_bi.pkl     | 0.833300           | 0.830788           |
| mnb_uni_ns.pkl     | 0.805863           | 0.804662           |
| mnb_bi_ns.pkl      | 0.780450           | 0.780288           |
| mnb_uni_bi_ns.pkl  | 0.823575           | 0.822863           |
|--------------------------------------------------------------|
```
```
# Test Accuracy with Best Model for each Text Feature type with stop words and stop words remove. 
|--------------------------------------------------------------|
| Stopwords removed  | Text features      | Accuracy (test set)|
|--------------------------------------------------------------|
| yes                | unigrams           | 0.804662           |
| yes                | bigrams            | 0.780288           |
| yes                | unigrams+bigrams   | 0.822863           |
| no                 | unigrams           | 0.807388           |
| no                 | bigrams            | 0.824887           |
| no                 | unigrams+bigrams   | 0.830788           |
|--------------------------------------------------------------|
```
##----------------------------------------------------------------------------------------------------------------------------------------
## Which condition performed better: with or without stopwords? Write a brief paragraph (5-6 sentences) discussing why you think there is a difference in performance.
##----------------------------------------------------------------------------------------------------------------------------------------
The comparison of model performance between conditions with and without stopwords reveals that including stopwords results in better performance. The highest test accuracy observed with stopwords was 0.830788 for the unigrams+bigrams model, while the highest without stopwords was 0.822863. Despite being common and often considered non-informative, stopwords play a vital role in preserving the syntactic and semantic structure of sentences. They contribute to the contextual understanding of the text by providing linkage and meaning to content words. The removal of stopwords might lead to a loss of this structure and context, making it more challenging for the classifier to accurately classify sentiments.

##----------------------------------------------------------------------------------------------------------------------------------------
## Which condition performed better: unigrams, bigrams or unigrams+bigrams? Briefly (in 5-6 sentences) discuss why you think there is a difference?
##----------------------------------------------------------------------------------------------------------------------------------------
The condition with unigrams+bigrams outperformed the conditions with only unigrams or bigrams. For instance, the unigrams+bigrams model achieved a test accuracy of 0.830788 with stopwords and 0.822863 without stopwords, both of which are higher than the accuracies for unigrams or bigrams alone. Bigrams alone can sometimes overlook details in single words, and unigrams alone might miss contextual dependencies between words. Unigrams capture individual word frequencies, which are crucial for sentiment analysis, while bigrams capture the sequences and context between words, adding more depth to the feature set. By combining unigrams and bigrams, a richer feature set is provided that captures both individual word occurrences and word pair sequences, thereby enhancing the model’s ability to understand and classify text accurately.
