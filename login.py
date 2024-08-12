import streamlit as st

# Login function
def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()


# Show login page
def show_login_page():
    st.title("Log in")
    st.write("Please log in to access the app.")
    login()

