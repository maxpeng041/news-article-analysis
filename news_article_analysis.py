#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.metrics import *
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tag import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import utils

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from itertools import islice
from tqdm import tqdm
import multiprocessing
import collections
import random
import string

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 50)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


# ### First look at the data

# In[2]:


path = './Articles'
files = glob.glob(path+'/*.csv')

li = []

for file in files:
    df = pd.read_csv(file, index_col=None, header=0)
    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)
data = data.drop(data.columns[0], axis=1)


# In[3]:


data.sample(10)


# In[4]:


data.info()


# In[5]:


data['title'] = data['title'].fillna('')


# ### Find articles about Hillary Clinton published by Washington Post and New York Times

# In[6]:


clinton_data = data[(data['year'] == 2016) &
                    (data['title'].str.contains('Hillary Clinton'))]
clinton_data = clinton_data[(clinton_data['publication'] == 'Washington Post') | 
                            (clinton_data['publication'] == 'New York Times')]
clinton_data.info()


# In[7]:


pd.set_option('display.max_colwidth', 500)
clinton_data[['title', 'publication', 'author']].sample(5).reset_index()


# In[8]:


pd.set_option('display.max_colwidth', 50)


# ### Manually label authors' gender (B indicate the authors of an article have different genders)

# In[9]:


clinton_data_gender = pd.read_csv('clinton_data_gender.csv', index_col=None, header=0)
clinton_data_gender.sample(5)


# In[10]:


clinton_data_gender.info()


# In[11]:


clinton_data_gender['gender'].value_counts()


# In[12]:


clinton_data_binary = clinton_data_gender[clinton_data_gender['gender']!='B']


# ### Preprocess the text

# In[13]:


# some preprocessing is done after pos tagging to improve the tag quality
def clean_text(unprocessed_string):
    cleaned_text = ""
    unprocessed_string = unprocessed_string.lower()
    unprocessed_string = unprocessed_string.replace("'", "").replace(".", "").replace("_", "")

    text_tokens = word_tokenize(unprocessed_string)
    for word in text_tokens:
        if word not in string.punctuation:
            #if word not in stopword_list:
                if len(word) > 1:
                    cleaned_text = cleaned_text + " " + word
    cleaned_text = ("").join(cleaned_text)
    return cleaned_text

stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                 "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
                 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
                 "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'own', 
                 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 
                 "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']


# In[14]:


X = clinton_data_binary['content'].reset_index(drop=True).copy()
y = clinton_data_binary['gender'].reset_index(drop=True).copy()

for i in range(len(X)):
    X[i] = clean_text(X[i])


# ### Compare models

# In[15]:


def print_scores(scores):
    k = len(scores['test_precision_macro'])
    print('precision_macro:    ' + str(sum(scores['test_precision_macro']) / k))
    print('recall_macro:       ' + str(sum(scores['test_recall_macro']) / k))
    print('f1_macro:           ' + str(sum(scores['test_f1_macro']) / k))
    print('precision_weighted: ' + str(sum(scores['test_precision_weighted']) / k))
    print('recall_weighted:    ' + str(sum(scores['test_recall_weighted']) / k))
    print('f1_weighted:        ' + str(sum(scores['test_f1_weighted']) / k))
    
scoring = ['precision_macro', 'recall_macro', 'f1_macro',
           'precision_weighted', 'recall_weighted', 'f1_weighted']


# #### Logistic Regression

# In[16]:


vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopword_list)
vectorizer.fit(X)
X_tfidf = vectorizer.transform(X)

lr_model = LogisticRegression()

scores = cross_validate(lr_model, X_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# #### Naive Bayes

# In[17]:


nb_model = MultinomialNB()

scores = cross_validate(nb_model, X_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# We can see logistic regression achieves better results, therefore we will use logistic regression for further exploration

# ### Logistic regression with TF-IDF + n-grams

# In[18]:


vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=1000, stop_words=stopword_list)
vectorizer.fit(X)
X_tfidf = vectorizer.transform(X)

lr_model = LogisticRegression(C=10000)

scores = cross_validate(lr_model, X_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# #### Most informative features from logistic regression

# In[19]:


lr_model = LogisticRegression(C=10000)
lr_model.fit(X_tfidf, y)

feature_names = vectorizer.get_feature_names() 
coefs_with_fns = sorted(zip(lr_model.coef_[0], feature_names)) 
coef_word=pd.DataFrame(coefs_with_fns)
coef_word.columns='coefficient','word'
most_pos = coef_word.sort_values(by='coefficient', ascending=True).head(20).reset_index(drop=True)
most_neg = coef_word.sort_values(by='coefficient', ascending=False).head(20).reset_index(drop=True)
pd.concat([most_pos, most_neg], axis=1)


# ### Create pos tagged content

# In[20]:


# get pos tags first, then remove stop words
# this ensures the tags are correct

X_pos = X.copy()
for i in range(len(X_pos)):
    article = X_pos[i]
    article = pos_tag(word_tokenize(article))
    article = [x for x in article if x[0] not in stopword_list]
    for j in range(len(article)):
        article[j] = article[j][0] + '-' + article[j][1]
    X_pos[i] = TreebankWordDetokenizer().detokenize(article)


# In[21]:


print(X_pos[0][:1000])


# ### Logistic regression with pos tagging + TF-IDF + n-grams

# In[22]:


X_pos_tag = X_pos
y = clinton_data_binary['gender']

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=1000, token_pattern=r"(?u)\b\w[\w-]*\w\b")
vectorizer.fit(X_pos_tag)
X_pos_tag_tfidf = vectorizer.transform(X_pos_tag)

lr_model = LogisticRegression(C=10000, max_iter=1000)

scores = cross_validate(lr_model, X_pos_tag_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# In[23]:


lr_model = LogisticRegression(C=10000, max_iter=1000)
lr_model.fit(X_pos_tag_tfidf, y)

feature_names = vectorizer.get_feature_names()
coefs_with_fns = sorted(zip(lr_model.coef_[0], feature_names)) 
coef_word=pd.DataFrame(coefs_with_fns)
coef_word.columns='coefficient','word'
most_pos = coef_word.sort_values(by='coefficient', ascending=True).head(20).reset_index(drop=True)
most_neg = coef_word.sort_values(by='coefficient', ascending=False).head(20).reset_index(drop=True)
pd.concat([most_pos, most_neg], axis=1)


# ### Doc2vec

# In[28]:


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

n_splits = 5
kf = KFold(n_splits=n_splits)
cv_precision_macro = []
cv_recall_macro = []
cv_f1_macro = []
cv_precision_weighted = []
cv_recall_weighted = []
cv_f1_weighted = []
featureset = clinton_data_binary[['content', 'gender']]

for train_index, test_index in kf.split(featureset):
    train = featureset.iloc[train_index]
    test = featureset.iloc[test_index]
    
    train_tagged = train.apply(lambda r: TaggedDocument(words=word_tokenize(r['content']), tags=[r.gender]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(words=word_tokenize(r['content']), tags=[r.gender]), axis=1)

    cores = multiprocessing.cpu_count()
    model_dbow = Doc2Vec(dm=0, vector_size=100, negative=0, hs=0, min_count=2, sample=0, workers=cores)
    model_dbow.build_vocab([x for x in train_tagged.values])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in train_tagged.values]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    y_train, X_train = vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = vec_for_learning(model_dbow, test_tagged)
    lr_model = LogisticRegression(n_jobs=1, C=1e5, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    cv_precision_macro.append(precision_score(y_test, y_pred, average='macro'))
    cv_recall_macro.append(recall_score(y_test, y_pred, average='macro'))
    cv_f1_macro.append(f1_score(y_test, y_pred, average='macro'))
    
    cv_precision_weighted.append(precision_score(y_test, y_pred, average='weighted'))
    cv_recall_weighted.append(recall_score(y_test, y_pred, average='weighted'))
    cv_f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))

avg_precision_macro = sum(cv_precision_macro)/n_splits
avg_recall_macro = sum(cv_recall_macro)/n_splits
avg_f1_macro = sum(cv_f1_macro)/n_splits

avg_precision_weighted = sum(cv_precision_weighted)/n_splits
avg_recall_weighted = sum(cv_recall_weighted)/n_splits
avg_f1_weighted = sum(cv_f1_weighted)/n_splits

print('precision_macro:    ' + str(avg_precision_macro))
print('recall_macro:       ' + str(avg_recall_macro))
print('f1_macro:           ' + str(avg_f1_macro))
print('precision_weighted: ' + str(avg_precision_weighted))
print('recall_weighted:    ' + str(avg_recall_weighted))
print('f1_weighted:        ' + str(avg_f1_weighted))

