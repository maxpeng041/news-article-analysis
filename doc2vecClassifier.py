import string
import pandas as pd
import numpy as np
import train as train
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.metrics import *
import random
from sklearn.model_selection import KFold, train_test_split
import collections
from gensim.models import Doc2Vec
import re
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn import utils
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('clinton_data_gender.csv')
print(df['gender'].value_counts())

df = df[df['gender'] != 'B']
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

for row in df['content']:
    clean_text(row)

train, test = train_test_split(df, test_size=0.3, random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r.gender]), axis=1)
test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r.gender]), axis=1)
import multiprocessing
cores = multiprocessing.cpu_count()


model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, classification_report

detailedReport = classification_report(y_test, y_pred)

print("Below is a detailed report of the models performance")
print(detailedReport)
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
