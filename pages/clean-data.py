import streamlit as st
from menu import menu_with_redirect
from st_aggrid import AgGrid

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

st.title("Clean your dataset")
# st.markdown(f"You are currently logged with the role of {st.session_state.step}.")

# Component de streamlit pour afficher dataset 
AgGrid(st.session_state.dataset)