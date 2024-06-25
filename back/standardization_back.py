import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import seaborn as sns

def deleteNoneNumberColumns(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    return df[numeric_columns]

def encode_string_columns(df):
    dataFrame = df.copy()
    le = pp.LabelEncoder()
    for column in dataFrame.columns:
        if dataFrame[column].dtype == 'object':
            dataFrame[column] = le.fit_transform(dataFrame[column])
    return dataFrame

def min_max_standardization(df):
    df = encode_string_columns(df)
    scaler = pp.MinMaxScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

def z_score_standardization(df):
    df = encode_string_columns(df)
    return (df - df.mean()) / df.std()

def robust_standardization(df):
    df = encode_string_columns(df)
    scaler = pp.RobustScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

def hist_plot():
    # Obtenir la liste des caractéristiques
    caracteristiques = st.session_state.df_normalized.columns.tolist()

    # Créer un widget pour sélectionner la caractéristique
    caracteristique_selectionnee = st.selectbox('Sélectionnez une caractéristique', caracteristiques)

    # Afficher la distribution de la caractéristique sélectionnée
    fig, ax = plt.subplots()
    sns.histplot(st.session_state.df_normalized[caracteristique_selectionnee], kde=True, ax=ax)
    ax.set_xlabel(caracteristique_selectionnee)
    ax.set_ylabel('Fréquence')
    st.pyplot(fig)

def box_plot():
    df = st.session_state.df_normalized
    fig, ax = plt.subplots()
    ax.boxplot(df.values, labels=df.columns)
    ax.set_ylabel('Standardized values')
    ax.set_title('Boxplot of standardized data')
    st.pyplot(fig)
    
