import os
import sys
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


from collections import defaultdict
from pathlib import Path

import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing


from transformers import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

path= "/Users/gkbytes/FakeNewsNet-master/"


def read_data(id, secpath):
	dictionary={}
	df= pd.DataFrame()
	count=0

	for name in id:
		try:
			with  open(path+secpath+name+"/news content.json") as f:
				data= json.load(f)
				count=count+1
				#print(data)
				dictionary[name] = [data['url'], data['text']]
		except IOError:
			pass

		uid= list(dictionary.keys())
		values= list(dictionary.values())

		url= [i[0] for i in values]
		text= [i[1] for i in values]

		data_tuples = list(zip(uid,url,text))
		df= pd.DataFrame(data_tuples, columns=['uid','url','text'])

	return df


def tokens(df):
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	kk = []
	for text in df.text:
		templist=tokenizer.encode(text)
		kk.append(templist.tokens)
	df.text = kk
	return df

def removestopwords(text):
	x=[]
	for tokens in text:
		x.append([word for word in tokens if word not in stopwords.words('english')])
	return x

def getrep(df,rep):
	def dummy_fun(text):
		return text

	if rep == 'binary':
	    vectorizer = CountVectorizer(binary = True, analyzer='word',tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
	if rep == 'freq': #for freq
	    vectorizer = CountVectorizer(binary= False,analyzer='word',tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
	if rep == 'tfidf':
	    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None)
	if rep == 'ngramfreq':
		vectorizer= CountVectorizer(binary=False,analyzer='word', tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None,ngram_range=(1,2))
	if rep == 'ngrambin':
		vectorizer= CountVectorizer(binary=True,analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None,ngram_range=(1,2))
	if rep == 'ngramtfidf':
		vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None, ngram_range=(1,2))

	vectorizer.fit(df.text)
	vocab= vectorizer.vocabulary_
	train_vector= vectorizer.transform(df.text) #matrix of dimension (n,|Vocab|)

	return train_vector,vectorizer



def main():
	os.chdir(path)

	politifact_real= pd.read_csv(path + "dataset/politifact_real.csv")
	politifact_fake= pd.read_csv(path + "dataset/politifact_fake.csv")

	real_id= politifact_real.id
	fake_id = politifact_fake.id


	
	politifact_realdata= read_data(real_id,'code/fakenewsnet_dataset/politifact/real/')
	politifact_fakedata = read_data(fake_id,'code/fakenewsnet_dataset/politifact/fake/')

	print(politifact_realdata.shape)
	print(politifact_fakedata.shape)

	frames=[politifact_realdata,politifact_fakedata]
	politifact= pd.concat(frames)
	y= list(np.zeros(politifact_realdata.shape[0])) + list(np.ones(politifact_fakedata.shape[0]))

	politifact['label'] = y

	#tokenization
	cleaned_politifact = tokens(politifact)

	cleaned_politifact.text = removestopwords(cleaned_politifact.text)

	#vectorizations
	tfidf,tfidfvectzr= getrep(cleaned_politifact,'tfidf')
	freq, freqvectzr= getrep(cleaned_politifact,'freq')
	binary,binaryvectzr= getrep(cleaned_politifact,'binary')

	bigramfreq,bigram_freq_vectzr= getrep(cleaned_politifact, 'ngramfreq')
	bigrambinary, bigram_bin_vectzr= getrep(cleaned_politifact, 'ngrambin')
	bigramtfidf, bigram_tfidf_vectzr = getrep(cleaned_politifact, 'ngramtfidf')

	# obj=sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

if __name__ == "__main__":
    main()




























