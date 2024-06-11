import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from back.standardization_back import min_max_standardization, z_score_standardization, robust_standardization, hist_plot, box_plot

st.header('Data standardization')

def min_max():
    st.subheader('Min-Max Normalization')
    df_cleaned = st.session_state.df_cleaned
    if df_cleaned is not None:
        st.write('Cleaned DataFrame')
        AgGrid(df_cleaned)
        st.write('DataFrame after Min-Max Normalization')
        df_normalized = min_max_standardization(df_cleaned)
        AgGrid(df_normalized)
        st.session_state.df_normalized = df_normalized
        
def z_score():
    st.subheader('Z-Score Normalization')
    df_cleaned = st.session_state.df_cleaned
    if df_cleaned is not None:
        st.write('Cleaned DataFrame')
        AgGrid(df_cleaned)
        st.write('DataFrame after Z-Score Normalization')
        df_normalized = z_score_standardization(df_cleaned)
        AgGrid(df_normalized)
        st.session_state.df_normalized = df_normalized

def robust():
    st.subheader('Robust Normalization')
    df_cleaned = st.session_state.df_cleaned
    if df_cleaned is not None:
        st.write('Cleaned DataFrame')
        AgGrid(df_cleaned)
        st.write('DataFrame after Robust Normalization')
        df_normalized = robust_standardization(df_cleaned)
        AgGrid(df_normalized)
        st.session_state.df_normalized = df_normalized

if 'df_cleaned' not in st.session_state:
    st.error('Please import and clean dataset before standardizing it.')
else:
    standardisation_method = st.radio('Choose a standardization method:', ('Min-Max', 'Z-Score', 'Robust'))
    if standardisation_method == 'Min-Max':
        min_max()
    elif standardisation_method == 'Z-Score':
        z_score()
    elif standardisation_method == 'Robust':
        robust()
    else:
        st.error('Invalid standardization method')
        exit()

    hist_plot('Histogram of standardized data')
    box_plot()
