import requests
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import *

ner_api = "https://api-inference.huggingface.co/models/Jean-Baptiste/camembert-ner"
ner_headers = {"Authorization": "Bearer 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"}
ts_api = "https://api-inference.huggingface.co/models/sshleifer/distilbart-xsum-12-3"
ts_headers = {"Authorization": "Bearer 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"}
tc_api = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
tc_headers = {"Authorization": "Bearer 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"}

def ts(payload):
	response = requests.post(ts_api, headers=ts_headers, json=payload)
	return response.json()
def ner(payload):
	response = requests.post(ner_api, headers=ner_headers, json=payload)
	return response.json()
def tc(payload):
    response = requests.post(tc_api, headers=tc_headers, json=payload)
    return response.json()

st.title('NLP Assist')
# file_up = st.file_uploader("Upload a file", type='csv')
file_up = 'assets/Precily_Text_Similarity.csv'
if file_up is not None:
    st.success("File uploaded successfully")
    df = pd.read_csv(file_up)
    df.dropna(inplace=True)
    obj = []
    for i in df.columns:
        type = df[i].dtypes
        if type == 'object':
            obj.append(i)
        else:pass

    st.header("Select Variables")
    selected = st.multiselect('Variables', obj)
    st.write(len(selected), 'variables selected')

    st.subheader('Exploratory Data Analysis on Selected Variables')
    eda_tasks = st.multiselect('EDA Tasks', 
                               ['N Words', 'Language Detection', 
                                'Subjectivity', 'Polarity', 
                                'Sentiment'])
    # n words
    if len(selected) < 1:
        df['n_words'] = df[selected].apply(lambda x: nWords(str(x)))
    else:
        for i in selected:
            df['n_words'+' '+str(i)] = df[i].apply(lambda x: nWords(str(x)))
    # Language Detection
    if len(selected) < 1:
        df['languages'] = df[selected].apply(lambda x: getLanguages(str(x)))
    else:
        for i in selected:
            df['languages'+' '+str(i)] = df[i].apply(lambda x: getLanguages(str(x)))
    # subjectivity
    if len(selected) < 1:
        df['subjectivity'] = df[selected].apply(lambda x: getSubjectivity(str(x)))
    else:
        for i in selected:
            df['subjectivity'+' '+str(i)] = df[i].apply(lambda x: getSubjectivity(str(x)))
    # polarity
    if len(selected) < 1:
        df['polarity'] = df[selected].apply(lambda x: getPolarity(str(x)))
    else:
        for i in selected:
            df['polarity'+' '+str(i)] = df[i].apply(lambda x: getPolarity(str(x)))
    # Sentiment
    if len(selected) < 1:
        df['sentiment'] = df['polarity'].apply(getSentiment)
    else:
        for i in selected:
            df['sentiment'+' '+str(i)] = df['polarity'+' '+str(i)].apply(getSentiment)
    
    # Display Dataframe        
    button = st.radio('', ('👀 View Complete Dataframe', '👀 View Sample Dataframe'), 
                      horizontal=True,
                      label_visibility='collapsed')
    if button == '👀 View Complete Dataframe':
        st.dataframe(df)
    else:
        st.dataframe(df.sample(5))

    