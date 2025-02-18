## Add execution instructions here
## To activate the Environment first write the command 
## -------------------------------------------------------------------------------------
`source /home/gsahu/miniconda3/bin/activate /home/gsahu/miniconda3/envs/msci641_env/`

## To run the main.py file 
`python3 a3/main.py /DATA1/smturaju/assignments/a1/data`

## To Run the inference file 
## We have used genism library to train a Word2Vec model on the Amazon corpus
## Model name : w2v.model
`python3 a3/inference.py /DATA1/smturaju/assignments/a3/data/word.txt`

## Explanation of main.py
1. Read the pos.txt and neg.txt
2. Concatenate pos.txt and neg.txt 
3. Apply the preprocessing technique using 'gensim.utils.simple_preproces()' that converting the text to lowercase, tokenizing the sentence into individual words, removing punctuation
4. Apply the Word2Vec model

## Explanation of inference.py
1. In the word.txt file contains 'good', and 'bad' words 
2. Predict top 20 similar words for good and bad respectively

##----------------------------------------------------------------------------------------------------------------------------------------
## Find 20 most similar words to “good” and “bad”
##----------------------------------------------------------------------------------------------------------------------------------------

|--------------------------------|--------------------------------|
| Words most similar to 'good'   | Words most similar to 'bad'    |
|--------------------------------|--------------------------------|
| Word               | Score     | Word               | Score     |
|--------------------------------|--------------------------------|
| great              | 0.808     | horrible           | 0.663     |
| decent             | 0.799     | good               | 0.653     |
| nice               | 0.717     | awful              | 0.633     |
| fantastic          | 0.716     | terrible           | 0.613     |
| terrific           | 0.709     | obvious            | 0.589     |
| superb             | 0.704     | funny              | 0.573     |
| wonderful          | 0.702     | gross              | 0.561     |
| excellent          | 0.678     | poor               | 0.559     |
| bad                | 0.653     | sad                | 0.556     |
| impressive         | 0.651     | stupid             | 0.548     |
| amazing            | 0.650     | shabby             | 0.541     |
| awesome            | 0.642     | fake               | 0.533     |
| fabulous           | 0.635     | lazy               | 0.532     |
| poor               | 0.621     | strange            | 0.532     |
| perfect            | 0.610     | dumb               | 0.530     |
| okay               | 0.584     | harsh              | 0.525     |
| ok                 | 0.583     | weak               | 0.524     |
| outstanding        | 0.579     | lame               | 0.524     |
| terrible           | 0.576     | weird              | 0.516     |
| alright            | 0.569     | cheesy             | 0.516     |
|--------------------------------|--------------------------------|

#----------------------------------------------------------------------------------------------------------------------------------------
# Are the words most similar to “good” positive, and words most similar to “bad” negative? Why this is or isn’t the case? Explain your intuition briefly (in 5-6 sentences).
#----------------------------------------------------------------------------------------------------------------------------------------
Yes, the words most similar to “good” are generally positive, and the words most similar to “bad” are usually negative. This pattern occurs because word embeddings, like those produced by Word2Vec, capture the semantic relationships between words based on their usage in Amazon review text. Words frequently appearing in similar contexts tend to have similar meanings and are located near each other in the vector space. Positive words like "great," "fantastic," and "wonderful" are often used in the same context as "good," while negative words like "horrible," "awful," and "terrible" are used in the same context as "bad," reflecting their respective positive and negative meanings However, there are exceptions, such as “bad” appearing in the list of words similar to “good”. This could be due to phrases like “not good” where “good” and “bad” can appear in similar contexts. 
