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
uploaded_file = st.file_uploader("Choose a file", type="csv")

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


### 2. Data pre-processing and cleaning
st.subheader('2. Data Pre-processing and Cleaning')
def load_csv(file):
    try:
        df = pd.read_csv(file)
        return df
    except pd.errors.EmptyDataError:
        st.error('Le fichier CSV est vide.')
    except pd.errors.ParserError:
        st.error('Erreur lors de la lecture du fichier CSV.')
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue : {e}')
    return None

csv_file = st.file_uploader("Choose a CSV file to clean", type="csv")

if csv_file is not None:
    df = load_csv(csv_file)
    if df is not None:
      st.subheader('Original DataFrame')
      AgGrid(df, key="original_df")

      st.subheader('Summary Statistics')
      st.write(df.describe())

      st.subheader('Missing Values')
      st.write(df.isnull().sum())

      # Options to handle missing values
      st.subheader('Handle Missing Values')
      missing_value_action = st.radio(
        "Choose how to handle missing values:",
        ('Remove Rows', 'Remove Columns')
      )

      if missing_value_action == 'Remove Rows':
        df_cleaned = df.dropna()
        st.write("Rows with missing values removed")
      if missing_value_action == 'Remove Columns':
        df_cleaned = df.dropna(axis=1)
        st.write("Columns with missing values removed")

      st.subheader(f'DataFrame after handling missing values ({missing_value_action})')
      AgGrid(df_cleaned, key="cleaned_df")  # Affiche le DataFrame nettoyé avec une clé unique

      st.subheader(f'Summary Statistics after Cleaning ({missing_value_action})')
      st.write(df_cleaned.describe())

      st.subheader(f'Missing Values after Cleaning ({missing_value_action})')
      st.write(df_cleaned.isnull().sum())

      replace_option = st.selectbox(
        'Choose an option to replace missing values',
        ('Median', 'Mean', 'Mode')
      )

      if replace_option == 'Median':
        ## Options to replace missing values with median
        st.subheader('Replace Missing Values with Median')
        numeric_columns = df.select_dtypes(include=np.number).columns
        df_median = df.copy()
        df_median[numeric_columns] = df_median[numeric_columns].fillna(df_median[numeric_columns].median())
        st.write(df_median)
      elif replace_option == 'Mean':
        ## Options to replace missing values with mean
        st.subheader('Replace Missing Values with Mean')
        numeric_columns = df.select_dtypes(include=np.number).columns
        df_mean = df.copy()
        df_mean[numeric_columns] = df_mean[numeric_columns].fillna(df_mean[numeric_columns].mean())
        st.write(df_mean)
      else:
        ## Options to replace missing values with mode of the column
        st.subheader('Replace Missing Values with Mode')
        not_numeric_columns = df.select_dtypes(exclude=np.number).columns
        df_mode = df.copy()
        df_mode[not_numeric_columns] = df_mode[not_numeric_columns].fillna(df_mode[not_numeric_columns].mode().iloc[0])
        st.write(df_mode)

      

      

      

