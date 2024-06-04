import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid

st.set_page_config(layout="wide")

st.title('Web application interactive')

df = pd.read_csv('netflix.csv')

# Component de streamlit pour afficher dataset 
AgGrid(df)

# Résumé statistique 
st.write(df.shape)
st.write(df.columns, df.isnull().sum())

# st.line_chart(df, y=['release year'])

option = st.selectbox(
     'What is your favorite color?',
     ('Blue', 'Red', 'Green'))

st.write('Your favorite color is ', option)

### 1. Exploration des données
st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension.lower() == 'csv':
        df = pd.read_csv(uploaded_file)
        st.subheader('DataFrame')
        st.write(df)
        st.subheader('Descriptive Statistics')
        st.write(df.describe())
    else:
        st.error('☝️ Please upload a CSV file')

st.sidebar.header('Setting')

col1, col2, col3 = st.columns(3)

user_name = st.sidebar.text_input('What is your name?')
user_emoji = st.sidebar.selectbox('Choose an emoji', ['', '😄', '😆', '😊', '😍', '😴', '😕', '😱'])
user_food = st.sidebar.selectbox('What is your favorite food?', ['', 'Tom Yum Kung', 'Burrito', 'Lasagna', 'Hamburger', 'Pizza'])

with col1:
  if user_name != '':
    st.write(f'👋 Hello {user_name}!')
  else:
    st.write('👈  Please enter your **name**!')

with col2:
  if user_emoji != '':
    st.write(f'{user_emoji} is your favorite **emoji**!')
  else:
    st.write('👈 Please choose an **emoji**!')

with col3:
  if user_food != '':
    st.write(f'🍴 **{user_food}** is your favorite **food**!')
  else:
    st.write('👈 Please choose your favorite **food**!')
