import streamlit as st

# Logout function
def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

# Show logout page
def show_logout_page():
    st.title("Log out")
    st.write("You have been logged out.")
    logout()
