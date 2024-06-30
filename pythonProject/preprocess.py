import os
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Decontract words im -> i am; didnt -> did not
def decontracted(phrase):
    """
    We first define a function to expand the contracted phrase into normal words
    """

    phrase = re.sub(r"wont", "will not", phrase)
    phrase = re.sub(r"wouldnt", "would not", phrase)
    phrase = re.sub(r"shouldnt", "should not", phrase)
    phrase = re.sub(r"couldnt", "could not", phrase)
    phrase = re.sub(r"cudnt", "could not", phrase)
    phrase = re.sub(r"cant", "can not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"doesnt", "does not", phrase)
    phrase = re.sub(r"didnt", "did not", phrase)
    phrase = re.sub(r"wasnt", "was not", phrase)
    phrase = re.sub(r"werent", "were not", phrase)
    phrase = re.sub(r"havent", "have not", phrase)
    phrase = re.sub(r"hadnt", "had not", phrase)
    phrase = re.sub(r"neednt", "need not", phrase)
    phrase = re.sub(r"isnt", "is not", phrase)
    phrase = re.sub(r"arent", "are not", phrase)
    phrase = re.sub(r"hasnt", "are not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# Get rid of user handles, tags, link, punctuation
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    #     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = decontracted(text)
    return text


def preprocess():
    df = pd.read_csv("./out.csv")
    df = df.dropna(axis=0)
    df["Text"] = df["Text"].apply(lambda x: clean_text(x))
    labelencoder = LabelEncoder()
    df['label_enc'] = labelencoder.fit_transform(df['emotion'])
    df.rename(columns={'label': 'label_desc'}, inplace=True)
    df.rename(columns={'label_enc': 'label'}, inplace=True)
    train_text, val_text, train_labels, val_labels = train_test_split(df['Text'], df['label'],
                                                                      random_state=2021,
                                                                      test_size=0.1,
                                                                      stratify=df['label'])
    return train_text, val_text, train_labels, val_labels
