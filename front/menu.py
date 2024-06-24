import streamlit as st


def csvLoad_menu():
    st.sidebar.page_link("main.py", label="I. Load/Exploration data")
    st.sidebar.page_link("pages/clean-data.py", label="II. Clean data")
    st.sidebar.page_link("pages/standardization.py", label="III. Standardization")
    st.sidebar.page_link("pages/clustering.py", label="IV. Clustering")
    # if st.session_state.step in ["admin", "super-admin"]:
    #     st.sidebar.page_link("pages/admin.py", label="Manage users")
    #     st.sidebar.page_link(
    #         "pages/super-admin.py",
    #         label="Manage admin access",
    #         disabled=st.session_state.step != "super-admin",
    # )


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