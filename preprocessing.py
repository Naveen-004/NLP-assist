import re
import spacy
import pandas as pd
from textblob import TextBlob
from nltk import word_tokenize
from langdetect import detect_langs
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def textClean(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # removed @mentions
    text = re.sub(r'#','',text) # remove the hash tag
    text = re.sub(r'RT[\s]+','',text) # remove RT
    text = re.sub(r'https?:\/\/\S+','',text) # Remove the hyper link
    text = re.sub(r'&amp;','',text) # remove &amp;
    text = re.sub(r'\s{2,}',' ', text)  # remove extra spaces
    text = re.sub(r'^\s+','',text)  # remove starting extra spaces
    return text

def tokenize(df, column):
    tokens = [word_tokenize(item) for item in df[column]]
    df['tokens'] = tokens
    return df

def noStopWords(text):
    nlp = spacy.load('en_core_web_sm')
    sentence = nlp(text)
    text = [word.text.strip() for word in sentence if not word.is_stop and not word.is_punct]
    return text

def nWords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    word_count = len(doc.text.split())
    return word_count

def getLanguages(text):
    return str(detect_langs(text)).split(':')[0][1:]    

def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity  # type: ignore
def getPolarity(text):
  return TextBlob(text).sentiment.polarity  # type: ignore

def stemming(text):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def lemming(text):
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

def getSentiment(score):
    if score < 0:
      return "Negative"
    if score == 0:
      return "Neutral"
    else:
      return "Positive"

