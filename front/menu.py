import streamlit as st


def csvLoad_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("main.py", label="I. Load/Exploration data")
    st.sidebar.page_link("pages/clean-data.py", label="II. Clean data")
    # if st.session_state.step in ["admin", "super-admin"]:
    #     st.sidebar.page_link("pages/admin.py", label="Manage users")
    #     st.sidebar.page_link(
    #         "pages/super-admin.py",
    #         label="Manage admin access",
    #         disabled=st.session_state.step != "super-admin",
    # )


def csvNotLoad_menu():
    # Show a navigation menu for unauthenticated users
    st.sidebar.page_link("main.py", label="I. Load dataset")


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if "step" not in st.session_state or st.session_state.step is None:
        csvNotLoad_menu()
        return
    csvLoad_menu()


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    if "step" not in st.session_state or st.session_state.step is None:
        st.switch_page("main.py")
    menu()