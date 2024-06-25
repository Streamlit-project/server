import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from back.clean_data import KNN, linearRegression, mean, median, mode

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

### 2. Data pre-processing and cleaning
st.subheader('2. Data Pre-processing and Cleaning')

df = st.session_state.dataset

if df is not None:
    if df is not None:
      st.subheader('Original DataFrame')
      st.write(df)

      st.subheader('Summary Statistics')
      st.write(df.describe())

      st.subheader('Missing Values')
      df = df.replace('unknown', None)
      st.write(df.apply(lambda x: x.isin([None, np.nan]).sum()))

      st.header('Do you want to remove missing values ?')      
      st.session_state.option_missing_value = st.selectbox(
        'Select yes or no',
        (None, 'Yes', 'No'),
        index=(None,'Yes', 'No').index(st.session_state.option_missing_value),
        key="selectbox_missing_value"
      )

      if(st.session_state.option_missing_value == 'Yes'):
        # Options to handle missing values
        st.subheader('Handle Missing Values')
        st.session_state.missing_value_action = st.radio(
          "Choose how to handle missing values:",
          ('Remove Rows', 'Remove Columns'),
          index=('Remove Rows', 'Remove Columns').index(st.session_state.missing_value_action),
          key="selectbox_handle_missing_values"
        )

        if st.session_state.missing_value_action == 'Remove Rows':
          df_cleaned = df.dropna()
          st.write("Rows with missing values removed")
        if st.session_state.missing_value_action == 'Remove Columns':
          df_cleaned = df.dropna(axis=1)
          st.write("Columns with missing values removed")

        st.subheader(f'DataFrame after handling missing values ({st.session_state.missing_value_action})')
        st.write(df_cleaned)

        st.subheader(f'Summary Statistics after Cleaning ({st.session_state.missing_value_action})')
        st.write(df_cleaned.describe())

        st.subheader(f'Missing Values after Cleaning ({st.session_state.missing_value_action})')
        st.write(df_cleaned.apply(lambda x: x.isin(['unknown', None, np.nan]).sum()))
      else:
        # Si il ya des variables non numériques dans le dataset, faire un mode ou un kNN
        if(df.select_dtypes(exclude=np.number).shape[1] > 0):
          
          st.subheader('1. Replace non numeric missing Values ')   
          st.session_state.string_options = st.selectbox(
            '1. Do you want to replace non numeric missing values',
            ('Yes, with Mode', 'No'),
            index=('Yes, with Mode', 'No').index(st.session_state.string_options),
            key="selectbox_method_non_numeric_missing_value"
          )
          
          if(st.session_state.string_options == 'Yes, with Mode'):
            not_numeric_columns = df.select_dtypes(exclude=np.number).columns
            df = mode(df)
          else:
            st.write('No action taken to replace non numeric missing values')
        
        st.subheader('2. Replace numeric missing Values ')     
        st.session_state.replace_option = st.selectbox(
          'Choose an option to replace missing values',
          ('Median', 'Mean', 'k-Nearest Neighbors', 'Linear Regression'),
          index=('Median', 'Mean', 'k-Nearest Neighbors', 'Linear Regression').index(st.session_state.replace_option),
          key="selectbox_method_numeric_missing_value"
        )

    
        if st.session_state.replace_option == 'Median':
          ## Options to replace missing values with median
          st.subheader('Replace Missing Values with Median')
          df_cleaned = median(df) 
          st.write(df_cleaned)
        elif st.session_state.replace_option == 'Mean':
          ## Options to replace missing values with mean
          st.subheader('Replace Missing Values with Mean')
          df_cleaned = mean(df)
          st.write(df_cleaned)
        elif st.session_state.replace_option == 'k-Nearest Neighbors':
          st.subheader('Replace Missing Values with k-Nearest Neighbors')
          df_cleaned = KNN(df)
          st.write("Données après imputation des valeurs manquantes")
          st.write(df_cleaned)
        else:
          st.subheader('Replace Missing Values with Linear Regression for Multiple Target Columns')
          df_cleaned = linearRegression(df)
          st.write(df_cleaned)

    st.session_state.df_cleaned = df_cleaned
    