import streamlit as st
from login import show_login_page
from logout import show_logout_page
from show_program_description import show_program_description
from show_forecast_salary import show_forecast_salary



# Configure Streamlit navigation
st.set_page_config(page_title="Salary Forecasting App", layout="wide")

# # Initialize session state
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# login_page = st.Page(show_login_page, title="Log in", icon=":material/login:")
# logout_page = st.Page(show_logout_page, title="Log out", icon=":material/logout:")

description = st.Page(show_program_description, title="Инструкция", icon=":material/dashboard:", default=True)
forecast_salary = st.Page(show_forecast_salary, title="Прогноз зарплаты", icon=":material/search:")
# history = st.Page("tools/history.py", title="History", icon=":material/history:")

pg = st.navigation(
    {
        # "Account": [logout_page],
        "Описание": [description],
        "Инструменты": [forecast_salary],
    }
)

pg.run()

