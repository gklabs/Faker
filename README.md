# Towards an automated system to detect disinformation in political media
## Introduction
Any piece of information can be real or fake based on various factors. In this project we use Deep learning and NLP techniques on two benchmark datasets to classify a piece of text into real or fake.

Please look at the final report to see more information on the architecture and the performance metrics. 

## LIAR
LIAR dataset has 6 different classes from True to Pants fire.

## FakeNewsNet
FNN has two different classes. Real and Fake.

## Tokeniser
We use BERT tokeniser from pyTorch. We remove stop words for TF-IDF representation and keep stop words for bigram TF-IDF.

## Embedding
### For LIAR 
* TF-IDF for baseline- logistic regression
* Google News vector Word2vec 300 dimenstions for neural architectures

### For Fake News Net
* TF-IDF for baseline logistic regression and Feed Forward Neural Network

## Neural Architectures
* RNN
* Bi-RNN
* LSTM
* Bi-LSTM
* GRU
* Bi-GRU
## Experimental set up
  * For each of them we test for 25 epochs at 
  * learning rate= 0.001 
  * using Adam Optimizer with a 
  * batch size of 200.


