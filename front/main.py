################################
#####  Fonction Front-end  ######
################################

import streamlit as st
from menu import menu
import pandas as pd
from st_aggrid import AgGrid

# Initialize st.session_state.role and st.session_state.role to None
if "step" not in st.session_state:
    st.session_state.step = None
    st.session_state.dataset = None
    st.session_state.optionmissingvalue = None

def show_data():
    st.subheader('Dataset')
    AgGrid(st.session_state.dataset)
    st.subheader('Descriptive Statistics')
    st.write(st.session_state.dataset.describe())

### 1. Exploration des données
## Upload CSV
st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file", type="csv")

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
      

      

