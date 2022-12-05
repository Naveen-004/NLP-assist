import streamlit as st
import pandas as pd
from preprocessing import *

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

    with st.form(key='my_form'):
        with st.sidebar:
            st.sidebar.header('To remove irregularities click on Clean Text')
            submit = st.form_submit_button(label='Clean Text')

    if submit:
        for i in df.columns:
            type = df[i].dtypes
            if type == 'object':
                st.write(df[i].apply(textclean))

    st.sidebar.header("Select Variable")
    selected = st.sidebar.selectbox('Variables', obj)

    tasks = ['Tokenization', 'POS tag', 'Stemming', 'Lemming', 'Stop-word Removal']
    tasks_func = [tokenize(df, selected),\
                    df[selected].apply(nostopwords),\
                    df[selected].apply(stemming),\
                    df[selected].apply(lemming),\
                    df[selected].apply(nostopwords)]

    st.sidebar.header("Select NLP Task")
    task = st.sidebar.multiselect('Tasks', tasks)
    result = map(tasks, tasks_func)
    for i in result:
        st.write(i)
    