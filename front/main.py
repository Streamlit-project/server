import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from back.main import KNN, linearRegression, load_csv, mean, median, mode

from menu import menu
import pandas as pd
from st_aggrid import AgGrid

# Initialize st.session_state.role and st.session_state.role to None
if "step" not in st.session_state:
    st.session_state.step = None
    st.session_state.dataset = None

def show_data():
    st.subheader('Dataset')
    AgGrid(st.session_state.dataset)
    st.subheader('Descriptive Statistics')
    st.write(st.session_state.dataset.describe())

### 1. Exploration des données
## Upload CSV
st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension.lower() == 'csv':
        df = pd.read_csv(uploaded_file)
        st.session_state.step = "clean-data"
        st.session_state.dataset = df
    else:
        st.error('☝️ Please upload a CSV file')

## Show dataset
if st.session_state.dataset is not None:
    show_data()

menu()

### 2. Data pre-processing and cleaning
st.subheader('2. Data Pre-processing and Cleaning')

csv_file = st.file_uploader("Choose a CSV file to clean", type="csv")

if csv_file is not None:
    df = load_csv(csv_file)
    if df is not None:
      st.subheader('Original DataFrame')
      st.write(df)

      st.subheader('Summary Statistics')
      st.write(df.describe())

      st.subheader('Missing Values')
      df = df.replace('unknown', None)
      st.write(df.apply(lambda x: x.isin([None, np.nan]).sum()))

      st.header('Do you want to remove missing values ?')
      option = st.selectbox('Select yes or no', ('Yes', 'No'))

      if(option == 'Yes'):
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
        st.write(df_cleaned)

        st.subheader(f'Summary Statistics after Cleaning ({missing_value_action})')
        st.write(df_cleaned.describe())

        st.subheader(f'Missing Values after Cleaning ({missing_value_action})')
        st.write(df_cleaned.apply(lambda x: x.isin(['unknown', None, np.nan]).sum()))
      else:
        # Si il ya des varaibles non numériques dans le dataset, faire un mode ou un kNN
        if(df.select_dtypes(exclude=np.number).shape[1] > 0):
          string_options = st.selectbox(
            'Do you want to replace non numeric missing values',
            ('Yes, with Mode', 'No')
          )

          if(string_options == 'Yes, with Mode'):
            st.subheader('Replace Missing Values with Mode')
            not_numeric_columns = df.select_dtypes(exclude=np.number).columns
            df[not_numeric_columns] = df[not_numeric_columns].fillna(df[not_numeric_columns].mode().iloc[0])
          else:
            st.write('No action taken to replace non numeric missing values')
             
        replace_option = st.selectbox(
          'Choose an option to replace missing values',
          ('Median', 'Mean', 'k-Nearest Neighbors', 'Linear Regression')
        )

        if replace_option == 'Median':
          ## Options to replace missing values with median
          st.subheader('Replace Missing Values with Median')
          df_median = median(df) 
          st.write(df_median)
        elif replace_option == 'Mean':
          ## Options to replace missing values with mean
          st.subheader('Replace Missing Values with Mean')
          df_mean = mean(df)
          st.write(df_mean)
        elif replace_option == 'k-Nearest Neighbors':
          st.subheader('Replace Missing Values with k-Nearest Neighbors')
          data_imputed = KNN(df)
          st.write("Données après imputation des valeurs manquantes")
          st.write(data_imputed)
        else:
          st.subheader('Replace Missing Values with Linear Regression for Multiple Target Columns')
          df_linear = linearRegression(df)
          st.write(df_linear)