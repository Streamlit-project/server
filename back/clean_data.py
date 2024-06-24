################################
#####  Fonction Back-end  ######
################################
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import numpy as np

### 2. Data pre-processing and cleaning
def median(df):
    try:
        numeric_columns = df.select_dtypes(include=np.number).columns
        df_median = df.copy()
        df_median[numeric_columns] = df_median[numeric_columns].fillna(df_median[numeric_columns].median())
        return df_median
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue dans la fonction Median : {e}')

def mean(df):
    try:
        numeric_columns = df.select_dtypes(include=np.number).columns
        df_mean = df.copy()
        df_mean[numeric_columns] = df_mean[numeric_columns].fillna(df_mean[numeric_columns].mean())
        return df_mean
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue dans la fonction Mean : {e}')

def mode(df):
    try:
        not_numeric_columns = df.select_dtypes(exclude=np.number).columns
        df_mode = df.copy()
        df_mode[not_numeric_columns] = df_mode[not_numeric_columns].fillna(df_mode[not_numeric_columns].mode().iloc[0])
        return df_mode
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue dans la fonction Mode : {e}')

def KNN(df):
    try:
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
        return data_imputed
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue dans la fonction KNN : {e}')

def linearRegression(df):
    try:
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

            return df
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue dans la fonction Linear Regression : {e}')


