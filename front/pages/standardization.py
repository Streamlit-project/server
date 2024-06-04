import streamlit as st
from menu import menu_with_redirect
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from back.standardization_back import min_max_standardization

st.header('Data standardization')

def min_max():
    st.subheader('Min-Max Normalization')
    df = st.session_state.dataset
    if df is not None:
        st.write('Original DataFrame')
        AgGrid(df)
        st.write('DataFrame after Min-Max Normalization')
        # df_normalized = min_max_standardization(df)
        # AgGrid(df_normalized)
        
def z_score():
    st.subheader('Z-Score Normalization')
    df = st.session_state.dataset
    if df is not None:
        st.write('Original DataFrame')
        AgGrid(df)
        st.write('DataFrame after Z-Score Normalization')

standardisation_method = st.radio('Choose a standardization method:', ('Min-Max', 'Z-Score'))

if standardisation_method == 'Min-Max':
    min_max()
elif standardisation_method == 'Z-Score':
    z_score()
