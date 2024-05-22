import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid


st.title('Web application interactive')

df = pd.read_csv('netflix.csv')

# Component de streamlit pour afficher dataset 
AgGrid(df)

# Résumé statistique 
st.write(df.shape)
st.write(df.columns, df.isnull().sum())

# st.line_chart(df, y=['release year'])
