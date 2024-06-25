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

###
# Fonction for part 1 :
# Take a column of dataset
# Return number of null value (None, np.nan and unknonw)
###
def count_null_value(colonne):
    cpt = 0
    for i in colonne:
        if(i is None):
            cpt += 1
        elif(i is np.nan):
            cpt += 1
        elif (i == 'unknown'):
            cpt += 1
    return cpt

### 
# Fonction for part 1 :
# Take dataframe in param
# Display statistics for columns with object type
# Return empty
###
def show_statistics_for_string_value(df):
    df2 = [["Nom colonne", "Valeur unique", "Valeur null", "Mode"]]
        
    for column in df.columns:
        if df[column].dtype == object:
            # Valeur unique
            value_unique = len(df[column].unique())
            # Valeur null
            valeur_null_1 = count_null_value(df[column])
            valeur_null_2 = df[column].isnull().sum()
            total_valeur_null = valeur_null_1 + valeur_null_2
            # Mode
            mode = df[column].mode()[0]
            # Add to the dataframe
            df2.append([column, value_unique, total_valeur_null, mode])

    # CSS pour masquer les indices de ligne
    hide_table_row_index = """
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """

    # Injection du CSS avec Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(df2)

###
# Fonction for part 1 :
# No param
# Initialise session_state for better UI/UX
# No return
###
def setup_session_state():
    if "step" not in st.session_state:
        st.session_state.step = None
        st.session_state.dataset = None
    
    if "option_missing_value" not in st.session_state:
        st.session_state.option_missing_value = 'Yes'

    if "standardisation_method" not in st.session_state:
        st.session_state.standardisation_method = 'Min-Max'
        
    if "missing_value_action" not in st.session_state:
        st.session_state.missing_value_action = 'Remove Rows'
        
    if "string_options" not in st.session_state:
        st.session_state.string_options = 'Yes, with Mode'
        
    if "replace_option" not in st.session_state:
        st.session_state.replace_option = 'Median'

    if "df_normalized" not in st.session_state:
        st.session_state.df_normalized = None
        
    if "algorithme_for_dataset" not in st.session_state:
        st.session_state.algorithme_for_dataset = None