import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

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

def set_optionMissingvalue():
    print(st.session_state._option)

df = st.session_state.dataset 
if df is not None:
    st.subheader('Original DataFrame')
    AgGrid(df, key="original_df")

    st.subheader('Summary Statistics')
    st.write(df.describe())

    # Pré-traitement => Unknown = null
    st.subheader('Missing Values')
    df = df.replace('unknown', np.nan)
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
        [None, "Median", "Mean", "Mode"],
        key="_option",
        on_change=set_optionMissingvalue,
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
    elif replace_option == 'Mode':
        ## Options to replace missing values with mode of the column
        st.subheader('Replace Missing Values with Mode')
        not_numeric_columns = df.select_dtypes(exclude=np.number).columns
        df_mode = df.copy()
        df_mode[not_numeric_columns] = df_mode[not_numeric_columns].fillna(df_mode[not_numeric_columns].mode().iloc[0])
        st.write(df_mode)
    
    for colum in df.columns:
        fig, ax = plt.subplots()
        if colum != 'ratingLevel':
            ax.hist(df[colum], bins=20)
            st.pyplot(fig)
            st.write(colum)

    st.session_state.df_cleaned = df_cleaned