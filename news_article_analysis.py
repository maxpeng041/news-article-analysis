#!/usr/bin/env python
# coding: utf-8

# In[182]:


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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from itertools import islice

import collections
import random

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 50)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# In[4]:


path = './Articles'
files = glob.glob(path+'/*.csv')

li = []

for file in files:
    df = pd.read_csv(file, index_col=None, header=0)
    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)
data = data.drop(data.columns[0], axis=1)


# In[5]:


data.sample(10)


# In[6]:


data.info()


# In[7]:


data['title'] = data['title'].fillna('')


# In[8]:


clinton_data = data[(data['year'] == 2016) &
                    (data['title'].str.contains('Hillary Clinton'))]
clinton_data = clinton_data[(clinton_data['publication'] == 'Washington Post') | 
                            (clinton_data['publication'] == 'New York Times')]
clinton_data.info()


# In[9]:


pd.set_option('display.max_colwidth', 500)
clinton_data[['title', 'publication', 'author']].sample(5).reset_index()


# In[10]:


pd.set_option('display.max_colwidth', 50)


# In[11]:


# clinton_data_1 = clinton_data[0:110]
# clinton_data_2 = clinton_data[110:220]
# clinton_data_3 = clinton_data[220:]


# In[12]:


# clinton_data_1.to_csv('clinton_data_1.csv', index=False)
# clinton_data_2.to_csv('clinton_data_2.csv', index=False)
# clinton_data_3.to_csv('clinton_data_3.csv', index=False)


# In[13]:


# li = []
# clinton_data_1_gender = pd.read_csv('clinton_data_1_gender.csv', index_col=None, header=0)
# clinton_data_2_gender = pd.read_csv('clinton_data_2_gender.csv', index_col=None, header=0)
# clinton_data_3_gender = pd.read_csv('clinton_data_3_gender.csv', index_col=None, header=0)
# li.append(clinton_data_1_gender)
# li.append(clinton_data_2_gender)
# li.append(clinton_data_3_gender)

# clinton_data_gender = pd.concat(li, axis=0, ignore_index=True)


# In[256]:


clinton_data_gender = pd.read_csv('clinton_data_gender.csv', index_col=None, header=0)
clinton_data_gender.sample(5)


# In[257]:


clinton_data_gender.info()


# In[258]:


clinton_data_gender['gender'].value_counts()


# In[137]:


clinton_data_binary = clinton_data_gender[clinton_data_gender['gender']!='B']


# ### PMI

# In[138]:


stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
                  'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'own', 'same', 
                  'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 
                  'll', 'm', 'o', 're', 've', 'y', 'ma']


# In[139]:


all_words_sw = [word.lower()
                for review in clinton_data_gender['content']
                for word in RegexpTokenizer(r'\w+').tokenize(review)
                if not word.lower() in stopwords_list
               ]

finder = BigramCollocationFinder.from_words(all_words_sw)
bgm = BigramAssocMeasures()
score = bgm.mi_like  # metric options: pmi or mi_like
collocations = {'_'.join(bigram): pmi for bigram, pmi in finder.score_ngrams(score)}

list(islice(collocations.items(), 30)) # return word pairs with highest scores


# In[140]:


all_words_sw_m = [word.lower()
                for review in clinton_data_gender[clinton_data_gender['gender']=='M']['content']
                for word in RegexpTokenizer(r'\w+').tokenize(review)
                if not word.lower() in stopwords_list
               ]

finder = BigramCollocationFinder.from_words(all_words_sw_m)
bgm = BigramAssocMeasures()
score = bgm.mi_like  # metric options: pmi or mi_like
collocations = {'_'.join(bigram): pmi for bigram, pmi in finder.score_ngrams(score)}

list(islice(collocations.items(), 30)) # return word pairs with highest scores


# In[141]:


all_words_sw_f = [word.lower()
                for review in clinton_data_gender[clinton_data_gender['gender']=='F']['content']
                for word in RegexpTokenizer(r'\w+').tokenize(review)
                if not word.lower() in stopwords_list
               ]

finder = BigramCollocationFinder.from_words(all_words_sw_f)
bgm = BigramAssocMeasures()
score = bgm.mi_like  # metric options: pmi or mi_like
collocations = {'_'.join(bigram): pmi for bigram, pmi in finder.score_ngrams(score)}

list(islice(collocations.items(), 30)) # return word pairs with highest scores


# ### Pos tagging

# In[102]:


import string
def clean_text(unprocessed_string):
    stop_words = stopwords.words()
    cleaned_text = ""
    unprocessed_string = np.str.lower(unprocessed_string)
    unprocessed_string = np.str.replace(unprocessed_string, "'", "")

    text_tokens = word_tokenize(unprocessed_string)
    for word in text_tokens:
        if word not in string.punctuation:
            if word not in stop_words:
                if len(word) > 1:
                    cleaned_text = cleaned_text + " " + word
    cleaned_text = ("").join(cleaned_text)
    return cleaned_text


# In[237]:


article_contents = clinton_data_binary['content'].reset_index(drop=True).copy()

for i in range(len(article_contents)):
    processed_article = clean_text(article_contents[i])
    processed_article = pos_tag(word_tokenize(processed_article))
    for j in range(len(processed_article)):
        processed_article[j] = processed_article[j][0] + '-' + processed_article[j][1]
    article_contents[i] = TreebankWordDetokenizer().detokenize(processed_article)


# In[238]:


print(article_contents[0][:1000])


# ### Run logistic regression and naive bayes with TF-IDF

# In[47]:


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


# In[259]:


X = clinton_data_binary['content']
y = clinton_data_binary['gender']

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=10000)
vectorizer.fit(X)
X_tfidf = vectorizer.transform(X)

lr_model = LogisticRegression(C=10000)

scores = cross_validate(lr_model, X_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# In[260]:


nb_model = MultinomialNB(alpha=1)

scores = cross_validate(nb_model, X_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# In[261]:


lr_model = LogisticRegression(C=10000)
lr_model.fit(X_tfidf, y)

feature_names = vectorizer.get_feature_names() 
coefs_with_fns = sorted(zip(lr_model.coef_[0], feature_names)) 
coef_word=pd.DataFrame(coefs_with_fns)
coef_word.columns='coefficient','word'
most_pos = coef_word.sort_values(by='coefficient', ascending=True).head(10).reset_index(drop=True)
most_neg = coef_word.sort_values(by='coefficient', ascending=False).head(10).reset_index(drop=True)
pd.concat([most_pos, most_neg], axis=1)


# ### Logistic regression with pos tagging

# In[263]:


X_pos_tag = article_contents
y = clinton_data_binary['gender']

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=20000, token_pattern=r"(?u)\b\w[\w-]*\w\b")
vectorizer.fit(X_pos_tag)
X_pos_tag_tfidf = vectorizer.transform(X_pos_tag)

lr_model = LogisticRegression(C=10000)

scores = cross_validate(lr_model, X_pos_tag_tfidf, y, scoring=scoring, cv=5)
print_scores(scores)


# In[264]:


lr_model = LogisticRegression(C=10000)
lr_model.fit(X_pos_tag_tfidf, y)

feature_names = vectorizer.get_feature_names() 
coefs_with_fns = sorted(zip(lr_model.coef_[0], feature_names)) 
coef_word=pd.DataFrame(coefs_with_fns)
coef_word.columns='coefficient','word'
most_pos = coef_word.sort_values(by='coefficient', ascending=True).head(10).reset_index(drop=True)
most_neg = coef_word.sort_values(by='coefficient', ascending=False).head(10).reset_index(drop=True)
pd.concat([most_pos, most_neg], axis=1)


# In[175]:


# # cross validation
# n_splits = 10
# kf = KFold(n_splits=n_splits)
# cv_accuracy = []
# cv_precision = []
# cv_recall = []
# for train, test in kf.split(featuresets):
#     train_data = np.array(featuresets)[train]
#     test_data = np.array(featuresets)[test]
    
#     prediction = []
#     classifier = nltk.NaiveBayesClassifier.train(train_data)
    
#     for i in range(len(test_data)):
#         prediction.append(classifier.classify(test_data[i][0]))
    
#     cv_accuracy.append(accuracy([i[1] for i in test_data], prediction))
#     cv_precision.append(precision(set([i[1] for i in test_data]), set(prediction)))
#     cv_recall.append(recall(set([i[1] for i in test_data]), set(prediction)))
#     # note: in nltk package, accuracy takes lists as inputs, and precision recall take sets as inputs

# avg_accuracy = sum(cv_accuracy)/n_splits
# avg_precision = sum(cv_precision)/n_splits
# avg_recall = sum(cv_recall)/n_splits

# print(str(n_splits) + '-fold cross validation')
# print('Accuracy')
# for i in cv_accuracy:
#     print(i)
    
# print('Precision')
# for i in cv_precision:
#     print(i)
    
# print('Recall')
# for i in cv_recall:
#     print(i)
    
# print('Average accuracy: ' + str(avg_accuracy))
# print('Average precision: ' + str(avg_precision))
# print('Average recall: ' + str(avg_recall))

