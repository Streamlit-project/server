import streamlit as st


def csvLoad_menu():
    st.sidebar.page_link("main.py", label="I. Load/Exploration data")
    st.sidebar.page_link("pages/clean-data.py", label="II. Clean data")
    if st.session_state.option_missing_value != None :
        st.sidebar.page_link("pages/standardization.py", label="III. Standardization")
    if st.session_state.algorithme_for_dataset != None :
        if st.session_state.algorithme_for_dataset == 'Clustering' :
            st.sidebar.page_link("pages/clustering.py", label="IV. Clustering")
        if st.session_state.algorithme_for_dataset == 'Prediction' :
            st.sidebar.page_link("pages/prediction.py", label="IV. Prediction")


def csvNotLoad_menu():
    st.sidebar.page_link("main.py", label="I. Load dataset")


def menu():
    if "step" not in st.session_state or st.session_state.step is None:
        csvNotLoad_menu()
        return
    csvLoad_menu()


def menu_with_redirect():
    if "step" not in st.session_state or st.session_state.step is None:
        st.switch_page("main.py")
    menu()