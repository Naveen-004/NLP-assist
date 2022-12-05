import requests
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import *
from datapurifier import Nlpeda, NLAutoPurifier


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
file_up = st.file_uploader("Upload a file", type='csv')
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
    eda = Nlpeda(df, selected, analyse= 'basic')  
    st.write(eda.df)

    st.subheader('Text Preprocessing on Selected Variable') 
    pure = NLAutoPurifier(df, selected)
    st.write(pure)

    # compare original and purified data
    random = np.random.randint(0, len(pure))
    st.header('Original Data vs Purified Data')
    st.subheader('Original Data')
    st.write(df[selected][random])
    st.subheader('Purified Data')
    st.write(pure[selected][random])

    models = ['Named Entity Recognition', 'Text summarization', 'Text classification']
    st.header('Select Model')
    model = st.selectbox('Models', models)
    st.write('Selected Model ➡️ ', model)
    final_df = []
    st.subheader(model)
    for i in range(len(pure)):
        if model == 'Named Entity Recognition':
            payload = {"inputs": pure[selected][i]}
            output = ner(payload)
            final_df.append(output)
        elif model == 'Text summarization':
            payload = {"inputs": pure[selected][i]}
            output = ts(payload)
            final_df.append(output)
        elif model == 'Text classification':
            payload = {"inputs": pure[selected][i]}
            output = tc(payload)
            final_df.append(output)

    def convert(df):
        return df.to_csv().encode('utf-8')
    final_df = pd.DataFrame(final_df)
    st.dataframe(final_df)
    csv = convert(final_df)
    st.download_button(label='Download CSV',
                       data=csv,
                       file_name='output.csv',
                       mime='text/csv')
    

    # tasks = ['Tokenization', 'POS tag', 'Stemming', 'Lemming', 'Stop-word Removal']
    # tasks_func = [tokenize(df, selected),\
    #                 df[selected].apply(nostopwords),\
    #                 df[selected].apply(stemming),\
    #                 df[selected].apply(lemming),\
    #                 df[selected].apply(nostopwords)]

    # st.sidebar.header("Select NLP Task")
    # task = st.sidebar.multiselect('Tasks', tasks)
    # result = map(tasks, tasks_func)
    # for i in result:
    #     st.write(i)