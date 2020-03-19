import pandas as pd
import os

os.chdir("D:\\Spring 2020\\Project\\liar_dataset")
train = pd.read_csv('train.tsv', sep='\t')
validation = pd.read_csv('valid.tsv', sep='\t')
colnames = ['file', 'tag', 'news', 'topic', 'speaker', 'speaker_job', 'speaker_state','speaker_party', 'ch_false','ch_barelytrue','ch_halftrue','ch_mostlytrue','ch_pantsonfire', 'Spoke_at' ]
train.columns = colnames
validation.columns = colnames
train.drop('file', axis = 1, inplace = True)
validation.drop('file', axis = 1, inplace = True)
liardata = pd.concat([train,validation])
liardata.to_csv("training.csv") 

train.shape # 10239
validation.shape #1283


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])

X = train.news
y = train.tag

X_test = validation.news
y_test = validation.tag

logreg.fit(X, y)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))

