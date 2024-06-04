################################
#####  Fonction Front-end  #####
################################

import streamlit as st
from menu import menu
import pandas as pd
from st_aggrid import AgGrid
from back.main import my_fonction

# Initialize st.session_state.role and st.session_state.role to None
if "step" not in st.session_state:
    st.session_state.step = None
    st.session_state.dataset = None
    st.session_state.optionmissingvalue = None

# Show data in Grid
def show_data():
    st.subheader('1. Initial data exploration')
    st.write('You can see your dataset in a grid. ')
    AgGrid(st.session_state.dataset)
    st.write('Here is a summary of the statistics (number, mean...)')
    st.write(st.session_state.dataset.describe())
    st.write('Now you can go to the Clean data section for clean your dataset. ')

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
    
my_fonction()

menu()
      

      

