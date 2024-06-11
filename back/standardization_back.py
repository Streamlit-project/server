import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import scipy.stats

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

def hist_plot(title):
    df = st.session_state.df_normalized
    fig, ax = plt.subplots()
    for column in df.columns:
        if pd.isnull(df[column]).all():
            continue
        ax.hist(df[column], bins=50, alpha=0.5, label=column)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def box_plot():
    df = st.session_state.df_normalized
    fig, ax = plt.subplots()
    ax.boxplot(df.values, labels=df.columns)
    ax.set_ylabel('Standardized values')
    ax.set_title('Boxplot of standardized data')
    st.pyplot(fig)