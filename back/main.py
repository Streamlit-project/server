################################
#####  Fonction Back-end  ######
################################
import streamlit as st
import pandas as pd

### 2. Data pre-processing and cleaning

def load_csv(file):
    try:
        df = pd.read_csv(file)
        return df
    except pd.errors.EmptyDataError:
        st.error('Le fichier CSV est vide.')
    except pd.errors.ParserError:
        st.error('Erreur lors de la lecture du fichier CSV.')
    except Exception as e:
        st.error(f'Une erreur inattendue est survenue : {e}')
    return None

# def median():
