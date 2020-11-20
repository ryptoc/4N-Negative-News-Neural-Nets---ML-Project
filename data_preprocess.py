import pandas as pd
import numpy as np

# LOAD

train_non_adverse = pd.read_csv("nam.csv")
train_adverse = pd.read_csv("am.csv")

# PREPROCESS

train = pd.concat([train_non_adverse, train_adverse], ignore_index=True)

train_nam = train[train.label == "nam"]
train_am = train[train.label == "am"]
train_random = train[train.label == "random"]

train_filtered = pd.concat([train_nam, train_am, train_random], ignore_index=True)
train_filtered = train_filtered.dropna(subset=['title', 'article'])

train_data = pd.DataFrame([])
train_data['text'] = train_filtered[['title', 'article']].agg(' '.join, axis=1)
train_data['label'] = train_filtered.label.map(dict(am=1, nam=0, random=2))

# FUNCTIONS

def getData():
    return train_data.copy()


def getAm():
    td = getData()
    td = td[td['label'] == 1]
    return td

def getNam():
    td = getData()
    td = td[td['label'] == 0]
    return td
    
def getRandom():
    td = getData()
    td = td[td['label'] == 2]
    return td


def getTrainData(include_random=False, random_as_2=False, shuffle=False):
    td = getData()
    if not include_random:
        td = td.loc[td['label'] != 2]
    
    if not random_as_2:
        td.loc[td['label'] == 2, 'label'] = 0
    
    if shuffle:
        td = td.sample(frac=1)
    
    return td


def getFullRowByIndex(idx):
    return train_filtered.loc[idx]