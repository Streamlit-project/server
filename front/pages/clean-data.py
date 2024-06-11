import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np

from back.main import KNN, linearRegression, load_csv, mean, median, mode


# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

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