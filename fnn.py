import os
import sys
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import timeit
import fnn_read_data as rd

from collections import defaultdict
from pathlib import Path

import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import preprocessing


from transformers import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.model_selection import train_test_split,KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





def tokenise(df):
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	templist= list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t), df.text))
	# print(templist)
	df.text= templist
	return df

# def removestopwords(text):
# 	x=[]
# 	for tokens in text:
# 		x.append([word for word in tokens if word not in stopwords.words('english')])
# 	return x

def getrep(train,validation,test,rep):
	def dummy_fun(text):
		return text

	if rep == 'binary':
		# train = removestopwords(train)
		# validation = removestopwords(validation)
		# test = removestopwords(test)
		vectorizer = CountVectorizer(binary = True, stop_words=None,analyzer='word',tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
	if rep == 'freq': #for freq
		# train = removestopwords(train)
		# validation = removestopwords(validation)
		# test = removestopwords(test)
		vectorizer = CountVectorizer(binary= False,stop_words=None,analyzer='word',tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
	if rep == 'tfidf':
		# train = removestopwords(train)
		# validation = removestopwords(validation)
		# test = removestopwords(test)
		vectorizer = TfidfVectorizer(analyzer='word', stop_words=None,tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None)
	if rep == 'ngramfreq':
		vectorizer= CountVectorizer(binary=False,analyzer='word',stop_words=None ,tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None,ngram_range=(1,2))
	if rep == 'ngrambin':
		vectorizer= CountVectorizer(binary=True,analyzer='word', stop_words=None, tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None,ngram_range=(1,2))
	if rep == 'ngramtfidf':
		vectorizer = TfidfVectorizer(analyzer='word', stop_words=None,tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None, ngram_range=(1,2))

	vectorizer.fit(np.array(train))
	vocab= vectorizer.vocabulary_

	print(rep)
	print("vocabulary length: " +str(len(vocab)))

	train_vector= vectorizer.transform(np.array(train))#matrix of dimension (n,|Vocab|)
	val_vector= vectorizer.transform(np.array(validation))
	test_vector= vectorizer.transform(np.array(test))


	return train_vector,val_vector, test_vector

def traintestvalidate(df,y):
	#train, validation and test
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)

	#X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

	print("shapes of train validation and test" )
	print(X_train.shape)
	#print(X_val.shape)
	print(X_test.shape)

	return X_train, y_train, X_test,y_test

def baseline(Xtrain,ytrain,X_test,rep):
	X = Xtrain.text.values
	y = np.array(ytrain)

	Kf= KFold(n_splits=5, random_state=1, shuffle=True)

	for train_index, test_index in Kf.split(X):
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_val = X[train_index], X[test_index]
		y_train, y_val = y[train_index], y[test_index]
		print(X_train[:5])
		print(X_val[:5])

		X_train,X_val,X_test= getrep(X_train,X_val,X_test,rep)
		LRclf = LogisticRegression(random_state=1,solver='saga',penalty='l2').fit(X_train, y_train)
		y_lr_pred= LRclf.predict(X_val)

		'''
		NBclf = BernoulliNB().fit(X_train, y_train)
		y_nb_pred= NBclf.predict(X_val)


		svmclf1 = SVC(gamma='auto').fit(X_train, y_train)
		y_svm_pred= svmclf1.predict(X_val)
		'''
		print("=========================================")
		print(rep)
		print("accuracy")
		print(accuracy_score(y_val, y_lr_pred))#,accuracy_score(y_val, y_nb_pred), accuracy_score(y_val, y_svm_pred))
		print("precision")
		print(precision_score(y_val, y_lr_pred))#,precision_score(y_val, y_nb_pred), precision_score(y_val, y_svm_pred))
		print("Recall")
		print(recall_score(y_val, y_lr_pred))#,recall_score(y_val, y_nb_pred), recall_score(y_val, y_svm_pred))
		print("F1 score")
		print(f1_score(y_val, y_lr_pred))#,f1_score(y_val, y_nb_pred), f1_score(y_val, y_svm_pred))
		print(confusion_matrix(y_val, y_lr_pred))#,confusion_matrix(y_val, y_nb_pred), confusion_matrix(y_val, y_svm_pred))
		print("=========================================")

	return y_lr_pred #,y_nb_pred,y_svm_pred


def main():
	start = timeit.default_timer()

	politifact_real= pd.read_csv("/Users/gkbytes/FakeNewsNet-master/dataset/politifact_real.csv")
	politifact_fake= pd.read_csv("/Users/gkbytes/FakeNewsNet-master/dataset/politifact_fake.csv")

	real_id= politifact_real.id
	fake_id = politifact_fake.id


	politifact_realdata_obj= rd.read_data(real_id,'/Users/gkbytes/FakeNewsNet-master/code/fakenewsnet_dataset/politifact/real/')
	politifact_fakedata_obj = rd.read_data(fake_id,'/Users/gkbytes/FakeNewsNet-master/code/fakenewsnet_dataset/politifact/fake/')

	politifact_realdata_obj.getdata()
	politifact_fakedata_obj.getdata()


	frames=[politifact_realdata_obj.df,politifact_fakedata_obj.df]
	politifact= pd.concat(frames)
	y= list(np.zeros(politifact_realdata_obj.df.shape[0])) + list(np.ones(politifact_fakedata_obj.df.shape[0]))

	print(politifact.shape)
	politifact['label'] = y

	politifact.to_csv("/Users/gkbytes/FakeNewsNet-master/dataset/Read.csv")
	
	
	#tokenization
	cleaned_politifact = tokenise(politifact)

	
	cleaned_politifact.drop(columns= ['url','uid'], inplace=True)

	#train, test validation split
	X_train, y_train, X_test,y_test =traintestvalidate(cleaned_politifact, y)

	#newli= []
	#for text in X_val.text:
	#	newli.append(' '.join(text))

	#vectorizations
	Representations = ['tfidf','freq','binary','ngramfreq','ngramtfidf','ngrambin']
	for rep in Representations:
		#y_lr_pred, y_nb_pred,y_svm_pred =baseline(X_train, y_train,X_test, rep)
		y_lr_pred=baseline(X_train, y_train,X_test, rep)		
		#output= pd.DataFrame(list(zip(newli,y_val,y_svm_pred,y_nb_pred,y_lr_pred)), columns=["text","truth","svm","nb","lr"])
		#output.to_csv("/Users/gkbytes/FakeNewsNet-master/"+rep+".csv")


	stop = timeit.default_timer()
	print('Time: ', stop - start)
	

if __name__ == "__main__":
    main()




























