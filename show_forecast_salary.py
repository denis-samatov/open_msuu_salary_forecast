import streamlit as st
import logging

from typing import Dict, List
from tools.parser import get_areas, get_professional_roles

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_salary(salary: str) -> bool:
    try:
        int(salary)
        return True
    except ValueError:
        return False

@st.cache_resource
def load_data_once():
    areas = get_areas()
    professional_roles = get_professional_roles()
    return areas, professional_roles

def show_forecast_salary() -> None:
    st.title("Прогнозирование зарплат")
    st.sidebar.title("Настройки")

    # Fetch areas and professional roles
    areas, professional_roles = load_data_once()
    
    # Sidebar inputs for user settings
    selected_areas: List[str] = st.sidebar.multiselect("Выберите области", list(areas.keys()), default=["Москва"], placeholder="Выберите область")
    area_ids: List[int] = [areas[area] for area in selected_areas]

    text: str = st.sidebar.text_input("Название должности", placeholder="Введите должность ...")
    per_page: int = st.sidebar.number_input("Вакансий на страницу", min_value=50, max_value=150, value=100, placeholder="Введите число...")
    pages: int = st.sidebar.number_input("Количество страниц", min_value=1, max_value=5, value=3, placeholder="Введите число...")
    only_with_salary: bool = st.sidebar.checkbox("Только с зарплатой", value=True)
    exchanges: List[str] = st.sidebar.multiselect("Валюты", ["RUB", "USD", "EUR"], default=["RUB", "USD", "EUR"])
    selected_roles: List[str] = st.sidebar.multiselect("Профессиональные роли", list(professional_roles.keys()), placeholder="Выберите роль")
    professional_roles_ids: List[int] = [professional_roles[role] for role in selected_roles]

    search_params: Dict[str, List] = {
        'area': area_ids,
        'text': text,
        'per_page': per_page,
        'only_with_salary': only_with_salary,
        "exchanges": exchanges,
        "professional_roles": professional_roles_ids
    }

    # Initialize vacancy details if not already in session state
    if 'vacancy_details' not in st.session_state:
        st.session_state.vacancy_details = {
            'vacancy_title': '',
            'vacancy_professional_role': '',
            'vacancy_description': '',
            'vacancy_salary': '',
            'vacancy_experience': '',
            'vacancy_employment': '',
            'vacancy_schedule': ''
        }

    # Form for user to input vacancy details
    with st.expander("Показать/Скрыть форму вакансии", expanded=True):
        with st.form(key='vacancy_form'):
            st.text_input("Название должности", value=st.session_state.vacancy_details['vacancy_title'], key='vacancy_title', max_chars=100)
            st.selectbox(
                "Профессиональная роль",
                list(professional_roles.keys()),
                index=list(professional_roles.keys()).index(st.session_state.vacancy_details['vacancy_professional_role']) if st.session_state.vacancy_details['vacancy_professional_role'] else 0,
                key='vacancy_professional_role'
            )
            st.text_area("Описание работы и навыки", value=st.session_state.vacancy_details['vacancy_description'], key='vacancy_description')
            salary_input = st.text_input("Заработная плата", value=st.session_state.vacancy_details['vacancy_salary'], key='vacancy_salary', max_chars=20)

            # Check if the input is a valid integer
            if validate_salary(salary_input):
                st.session_state.vacancy_details['vacancy_salary'] = int(salary_input)
            else:
                st.error("Введите целое число в поле 'Заработная плата'.")
            
            st.text_input("Опыт работы", value=st.session_state.vacancy_details['vacancy_experience'], key='vacancy_experience', max_chars=20)
            st.text_input("Тип занятости", value=st.session_state.vacancy_details['vacancy_employment'], key='vacancy_employment', max_chars=30)
            st.text_input("График работы", value=st.session_state.vacancy_details['vacancy_schedule'], key='vacancy_schedule', max_chars=20)

            submit_button: bool = st.form_submit_button('Векторизовать вакансию и спрогнозировать зарплату')
            logging.info(f"Submit button state: {submit_button}")

    # Process the submitted vacancy form
    st.session_state.vacancy_details = {
        'vacancy_title': st.session_state.vacancy_title,
        'vacancy_professional_role': st.session_state.vacancy_professional_role,
        'vacancy_description': st.session_state.vacancy_description,
        'vacancy_salary': st.session_state.vacancy_salary,
        'vacancy_experience': st.session_state.vacancy_experience,
        'vacancy_employment': st.session_state.vacancy_employment,
        'vacancy_schedule': st.session_state.vacancy_schedule
    }
    vacancy_details: str = f"""
        Название вакансии: {st.session_state.vacancy_title}.
        Профессиональная роль: {st.session_state.vacancy_professional_role}.
        Описание вакансии и навыки: {st.session_state.vacancy_description}
        Опыт работы: {st.session_state.vacancy_experience}.
        Тип занятости: {st.session_state.vacancy_employment}.
        График работы: {st.session_state.vacancy_schedule}.
    """
    logging.info(f"Vacancy details: {vacancy_details}")
