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

from back.main import load_csv

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
      st.write(df.isnull().sum())

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
        st.write(df_cleaned.isnull().sum())
      else:
        # Si il ya des varaibles non numériques dans le dataset, faire un mode ou un kNN
        if(df.select_dtypes(exclude=np.number).shape[1] > 0):
          string_options = st.selectbox(
            'Choose an option to replace non numeric missing values',
            ('Mode', 'k-Nearest Neighbors')
          )

          # if(string_options == 'Mode'):
          #   st.subheader('Replace Missing Values with Mode')
          #   not_numeric_columns = df.select_dtypes(exclude=np.number).columns
          #   df_mode = df.copy()
          #   df_mode[not_numeric_columns] = df_mode[not_numeric_columns].fillna(df_mode[not_numeric_columns].mode().iloc[0])
          #   st.write(df_mode)
          # else:
             

        replace_option = st.selectbox(
          'Choose an option to replace missing values',
          ('Median', 'Mean', 'Mode', 'k-Nearest Neighbors', 'Linear Regression')
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
        elif replace_option == 'k-Nearest Neighbors':
          ## Impute missing values with k-Nearest Neighbors
          st.subheader('Replace Missing Values with k-Nearest Neighbors')

          # Séparer les colonnes numériques et non numériques
          numeric_cols = df.select_dtypes(include=['number']).columns
          non_numeric_cols = df.select_dtypes(exclude=['number']).columns

          numeric_data = df[numeric_cols]
          non_numeric_data = df[non_numeric_cols]

          # Imputer les valeurs manquantes uniquement sur les colonnes numériques
          imputer = KNNImputer(n_neighbors=5)
          numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_cols)

          # Réassembler le dataset avec les colonnes non numériques
          data_imputed = pd.concat([numeric_data_imputed, non_numeric_data], axis=1)

          # Afficher les données après imputation
          st.write("Données après imputation des valeurs manquantes")
          st.write(data_imputed)
        else:
          st.subheader('Replace Missing Values with Linear Regression for Multiple Target Columns')

          feature_columns = st.multiselect('Select feature columns for imputation', df.select_dtypes(include=['number']).columns)
          target_columns = st.multiselect('Select target columns with missing values', df.select_dtypes(include=['number']).columns)

          if len(target_columns) == 0:
              st.warning("No target columns selected.")
          elif len(feature_columns) == 0:
              st.warning("No feature columns selected.")
          else:
              X = df[feature_columns]
              y = df[target_columns]

              imputer = SimpleImputer(strategy='mean')
              regressor = LinearRegression()
              pipeline = Pipeline([('imputer', imputer), ('regressor', regressor)])

              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

              # Appliquer l'imputation sur les colonnes cibles
              for col in target_columns:
                  if pd.isna(y[col]).any():  # Vérifier si des valeurs manquantes existent
                      y_train_col = SimpleImputer(strategy='mean').fit_transform(y_train[col].values.reshape(-1, 1)).ravel()
                      pipeline.fit(X_train, y_train_col)
                      missing_indices = np.where(np.isnan(y[col]))[0]
                      X_missing = X.iloc[missing_indices]
                      y_pred = pipeline.predict(X_missing)
                      df.loc[missing_indices, col] = y_pred
                  else:
                      st.warning(f"Aucune valeur manquante trouvée dans {col}. Pas besoin d'imputer.")

              st.write(df)