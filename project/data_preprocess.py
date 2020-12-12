import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

# LOAD

train_non_adverse = pd.read_csv("data/nam.csv")
train_adverse = pd.read_csv("data/am.csv")
train_random_additional = pd.read_csv("data/random.csv")
am_additional = pd.read_csv("data/am_additional.csv")
test = pd.read_csv("data/test.csv", usecols=["title", "article", "label"])

# PREPROCESS

train_concat = pd.concat([train_non_adverse, train_adverse, train_random_additional, am_additional], ignore_index=True)
train_filtered = train_concat[(train_concat.label == "nam") | (train_concat.label == "am") | (train_concat.label == "random")]
train_filtered = train_filtered.dropna(subset=['title', 'article'])

# FUNCTIONS

def getData():
    train_data = pd.DataFrame([])
    train_data['text'] = train_filtered.article
    train_data['label'] = train_filtered.label.map(dict(am=1, nam=0, random=2))

    return train_data.copy()

def get_n_sentences(n):
    train_data_sentences = pd.DataFrame([])
    train_data_sentences['text'] = train_filtered.article.apply(lambda a: " ".join(nltk.sent_tokenize(a)[0:n]))
    train_data_sentences['label'] = train_filtered.label.map(dict(am=1, nam=0, random=2))
    
    return train_data_sentences.copy()

def getTrainData(include_random=False, random_as_2=False, shuffle=False, no_title=False, n_sentences=-1):
    td = None
    if n_sentences >= 0:
        td = get_n_sentences(n_sentences)
    else:
        td = getData()
    
    if not no_title:
        if n_sentences == 0:
            td['text'] = train_filtered.title
        else:
            td['text'] = pd.DataFrame({ 'title': train_filtered.title, 'article': td.text }).agg('.\n'.join, axis=1)
    
    if not include_random:
        td = td.loc[td['label'] != 2]
    
    if not random_as_2:
        td.loc[td['label'] == 2, 'label'] = 0
    
    if shuffle:
        td = td.sample(frac=1)
    
    return td.copy()

def getTestData(no_title=False, n_sentences=-1):
    td = pd.DataFrame([])
    if n_sentences >= 0:
        td['text'] = test.article.apply(lambda a: " ".join(nltk.sent_tokenize(a)[0:n_sentences]))
    else:
        td['text'] = test.article
    
    td['label'] = test.label
    
    td = td.copy()
    
    if not no_title:
        if n_sentences == 0:
            td['text'] = test.title
        else:
            td['text'] = pd.DataFrame({ 'title': test.title, 'article': td.text }).agg('.\n'.join, axis=1)
    
    return td.copy()

def getFullRowByIndex(idx):
    return train_filtered.loc[idx]