import streamlit as st

# Accessing general configuration values
SEED = st.secrets["general"]["SEED"]
USD_TO_RUB = st.secrets["general"]["USD_TO_RUB"]
EUR_TO_RUB = st.secrets["general"]["EUR_TO_RUB"]
API_HH_AREAS = st.secrets["general"]["API_HH_AREAS"]
API_HH_PROFESSIONAL_ROLES = st.secrets["general"]["API_HH_PROFESSIONAL_ROLES"]
API_HH_VACANCIES = st.secrets["general"]["API_HH_VACANCIES"]

# Accessing API keys and sensitive information
API_HH_KEY = st.secrets["api_keys"]["API_HH_KEY"]
EMPLOYER_ID = st.secrets["api_keys"]["EMPLOYER_ID"]
