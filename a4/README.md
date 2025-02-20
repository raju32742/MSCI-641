# Add execution instructions here
# Add execution instructions here
# To activate the Environment first write the command 
# -------------------------------------------------------------------------------------
source /home/gsahu/miniconda3/bin/activate /home/gsahu/miniconda3/envs/msci641_env/

# To run the main.py file 
python3 a4/main.py /DATA1/smturaju/assignments/a1/data

# To Run the inference file 
# Model name : simpleNN model with relu, sigmoid and tanh activation function. 
python3 a4/inference.py /DATA1/smturaju/assignments/a4/data/sentence.txt relu 

#----------------------------------------------------------------------------------------------------------------------------------------
# Report your classification accuracy results in a table with three different activation functions in the hidden layer (ReLU, sigmoid and tanh). 
#----------------------------------------------------------------------------------------------------------------------------------------
# Model with StopWords
|--------------------------|----------|---------|--------|
| Regularization + Dropout |   ReLU   | Sigmoid |  Tanh  |
|--------------------------|----------|---------|--------|
| L2: 0, Dropout: 0        |  0.8201  |  0.8100 | 0.8103 |
| L2: 0.1, Dropout: 0      |  0.5001  |  0.4999 | 0.4999 |
| L2: 0.01, Dropout: 0     |  0.5001  |  0.5001 | 0.5001 |
| L2: 0.001, Dropout: 0    |  0.7821  |  0.7209 | 0.7831 |
|--------------------------|-----------------------------|
| L2: 0, Dropout: 0.2      |  0.8227  |  0.8117 | 0.8088 |
|--------------------------|-----------------------------|
| L2: 0.1, Dropout: 0.2    |  0.5001  |  0.5001 | 0.4999 |
| L2: 0.01, Dropout: 0.2   |  0.5001  |  0.4999 | 0.4999 |
| L2: 0.001, Dropout: 0.2  |  0.7830  |  0.4999 | 0.7842 |
| L2: 0, Dropout: 0.3      |  0.8219  |  0.8129 | 0.8090 |
| L2: 0.1, Dropout: 0.3    |  0.5001  |  0.5001 | 0.4999 |
| L2: 0.01, Dropout: 0.3   |  0.4999  |  0.5001 | 0.5001 |
| L2: 0.001, Dropout: 0.3  |  0.7806  |  0.4999 | 0.7828 |
| L2: 0, Dropout: 0.4      |  0.8207  |  0.8128 | 0.8089 |
| L2: 0.1, Dropout: 0.4    |  0.5001  |  0.4999 | 0.5001 |
| L2: 0.01, Dropout: 0.4   |  0.5001  |  0.4999 | 0.5001 |
| L2: 0.001, Dropout: 0.4  |  0.7827  |  0.4999 | 0.7820 |
| L2: 0, Dropout: 0.5      |  0.8188  |  0.8130 | 0.8099 |
| L2: 0.1, Dropout: 0.5    |  0.4999  |  0.4999 | 0.5001 |
| L2: 0.01, Dropout: 0.5   |  0.5001  |  0.4999 | 0.4999 |
| L2: 0.001, Dropout: 0.5  |  0.7763  |  0.5001 | 0.7811 |
|--------------------------|----------|---------|--------|

#----------------------------------------------------------------------------------------------------------------------------------------
# What effect do activation functions have on your results? What effect does addition of L2-norm regularization have on the results? What effect does dropout have on the results? Explain your intuitions briefly (up to 10 sentences)
#----------------------------------------------------------------------------------------------------------------------------------------
The activation functions (ReLU, Sigmoid, Tanh) play a crucial role in determining the output of the neural network. ReLU seems to perform better overall, likely due to its ability to deal with the vanishing gradient problem. Rule activation shows the best test accuracy=0.8227 with droupout rate = 0.2 and no regularization. 

L2-norm regularization is used to prevent overfitting by penalizing large weights. When L2 is 0.1, the accuracy drops significantly, indicating that the model might be too constrained and unable to learn effectively. However, smaller L2 values (0.01, 0.001) seem to provide a balance, preventing overfitting while still allowing the model to learn. When L2 = 0, the model provide the best estimated accuracy.

Dropout is another technique to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time. The results show that a moderate dropout rate (0.2-0.4) can improve the model’s performance, but too high a rate (0.5) starts to negatively impact the accuracy, possibly because it’s causing the model to underfit the data.

Overall, the shallow deep learning model with single hidden layer performs better with droupout and relu activation function. 