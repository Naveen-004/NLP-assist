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

    st.header("Select Variable")
    selected = st.selectbox('Variables', obj)
    st.write('Selected Variable ➡️ ', selected)

    st.subheader('Exploratory Data Analysis on Selected Variable')
    df['n_words'] = nWords(df[selected])
    view_button = st.button('👀View Dataframe', disabled=False)
    if view_button:
        st.dataframe(df)

    