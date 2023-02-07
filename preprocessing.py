import re
import spacy
from heapq import nlargest
from textblob import TextBlob
from string import punctuation
from langdetect import detect_langs
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS


# EDA
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

def getSentiment(score):
    if score < 0:
      return "Negative"
    if score == 0:
      return "Neutral"
    else:
      return "Positive"

# Text Preprocessing
def textClean(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # removed @mentions
    text = re.sub(r'#','',text) # remove the hash tag
    text = re.sub(r'RT[\s]+','',text) # remove RT
    text = re.sub(r'https?:\/\/\S+','',text) # Remove the hyper link
    text = re.sub(r'&amp;','',text) # remove &amp;
    text = re.sub(r'\s{2,}',' ', text)  # remove extra spaces
    text = re.sub(r'^\s+','',text)  # remove starting extra spaces
    return text

def tokenize(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.text for token in doc]

def stopWords(text):
    nlp = spacy.load('en_core_web_sm')
    sentence = nlp(text)
    text = [word.text.strip() for word in sentence if not word.is_stop and not word.is_punct]
    return text

def stemming(text):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text.split()]
    return text

def lemming(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.lemma_ for token in doc]

# Models
def ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return print([(entity.text, entity.label_) for entity in doc.ents])

punctuation = punctuation + ' '
def text_summarizer(text):
    nlp = spacy.load('en_core_web_sm')
    stopwords = list(STOP_WORDS)
    docx = nlp(text)

    # tokens and word frequency
    mytokens = [token.text for token in docx]
    word_frequencies = {}
    for word in docx:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    # maximum word frequency
    max_frequency = max(word_frequencies.values())

    # word frequency normalization
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    # sentence tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # sentence score and sorting
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    # nlargest
    select_length = int(len(sentence_list)*0.3)
    summary = nlargest(select_length, sentence_scores, 
                       key=sentence_scores.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary