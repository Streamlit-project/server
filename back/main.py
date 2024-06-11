################################
#####  Fonction Back-end  ######
################################

### IMPORT
import pandas as pd
import numpy as np
import streamlit as st 

### Fonction
def my_fonction():
    return "Ma fonction marche "

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
# Fonction for part 1 : Explorating data
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
