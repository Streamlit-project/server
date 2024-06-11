import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import scipy.stats

def deleteNoneNumberColumns(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    return df[numeric_columns]

def min_max_standardization(df):
    df = deleteNoneNumberColumns(df)
    scaler = pp.MinMaxScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

def z_score_standardization(df):
    df = deleteNoneNumberColumns(df)
    return (df - df.mean()) / df.std()