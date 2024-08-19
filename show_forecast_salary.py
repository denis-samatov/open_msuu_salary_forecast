import streamlit as st
import numpy as np
import requests
import logging
import pandas as pd
import plotly.graph_objects as go


from gensim.models.doc2vec import Doc2Vec
from typing import Dict, List, Tuple
from tools.api_links_and_constant import API_HH_AREAS, API_HH_PROFESSIONAL_ROLES
from tools.charts import plot_salary_distribution, plot_key_skills, plot_salary_boxplots, plot_professional_roles, plot_key_skills_wordcloud
from tools.parser import fetch_and_process_vacancies
from tools.create_dataset import create_dataset
from tools.process_vacancies import vectorize_vacancy
from tools.anomaly_detection import detect_anomalies
from tools.salary_forecasting import OPTICS_UMAP_Clusterer, get_salary_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch areas from the API
def get_areas() -> Dict[str, int]:
    response = requests.get(API_HH_AREAS)
    response.raise_for_status()
    areas_data = response.json()
    areas = {area['name']: area['id'] for country in areas_data for area in country['areas']}
    return areas

# Function to fetch professional roles from the API
def get_professional_roles() -> Dict[str, int]:
    response = requests.get(API_HH_PROFESSIONAL_ROLES)
    response.raise_for_status()
    roles_data = response.json()
    roles = {role['name']: role['id'] for category in roles_data['categories'] for role in category['roles']}
    return roles

def validate_salary(salary: str) -> bool:
    try:
        int(salary)  # Try converting the input to an integer
        return True
    except ValueError:
        return False

@st.cache_resource
def load_data_once():
    areas = get_areas()
    professional_roles = get_professional_roles()
    return areas, professional_roles

def show_forecast_salary() -> None:
    """
    Main function to display the salary forecasting Streamlit application. Handles user inputs,
    fetches vacancies, processes data, and displays analysis and forecasting results.
    """
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

    # Button to fetch vacancies based on user inputs
    if st.sidebar.button("Получить вакансии"):
        with st.spinner("Получение вакансий..."):
            df_vacancies = fetch_and_process_vacancies(search_params, pages=pages)
            st.success("Вакансии успешно получены.")
            st.session_state.df_vacancies = df_vacancies

    # Process vacancies if they exist in session state
    if 'df_vacancies' in st.session_state:
        if 'df_vacancies_hh' not in st.session_state:
            with st.spinner("Обработка набора данных..."):
                df_vacancies_hh, doc2vec_model = create_dataset(st.session_state.df_vacancies)
                st.success("Набор данных успешно обработан.")
                st.session_state.df_vacancies_hh = df_vacancies_hh
                st.session_state.doc2vec_model = doc2vec_model

        if 'cleaned_df' not in st.session_state:
            with st.spinner("Обнаружение аномалий..."):
                cleaned_df, anomalies_df = detect_anomalies(st.session_state.df_vacancies_hh)
                st.success("Аномалии обнаружены и удалены.")
                st.session_state.cleaned_df = cleaned_df

        # Display data analysis if cleaned data is available
        if 'cleaned_df' in st.session_state:
            st.header("Анализ данных")
            with st.expander("Показать/Скрыть анализ данных"):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_salary_dist = plot_salary_distribution(st.session_state.cleaned_df)
                        st.plotly_chart(fig_salary_dist, use_container_width=True)

                        fig_key_skills = plot_key_skills(st.session_state.cleaned_df)
                        st.plotly_chart(fig_key_skills, use_container_width=True)

                        fig_key_skills_wordcloud = plot_key_skills_wordcloud(st.session_state.cleaned_df)
                        st.plotly_chart(fig_key_skills_wordcloud, use_container_width=True)

                    with col2:
                        fig_salary_boxplot = plot_salary_boxplots(st.session_state.cleaned_df)
                        st.plotly_chart(fig_salary_boxplot, use_container_width=True)

                        fig_prof_roles = plot_professional_roles(st.session_state.cleaned_df)
                        st.plotly_chart(fig_prof_roles, use_container_width=True)


        st.header("Добавить вашу вакансию")

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
        if submit_button:
            logging.info("Form has been submitted.")
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

            # Vectorize the vacancy if the title and description are provided
            with st.spinner("Векторизация вакансии..."):
                if st.session_state.vacancy_details['vacancy_title'] and st.session_state.vacancy_details['vacancy_description']:
                    doc2vec_vacancy_vector: np.ndarray = vectorize_vacancy(st.session_state.vacancy_details['vacancy_title'], vacancy_details, st.session_state.doc2vec_model)
                    st.success("Вакансия успешно векторизована.")
                    st.session_state.doc2vec_vacancy_vector = doc2vec_vacancy_vector
                else:
                    st.error("Детали вакансии неполные. Пожалуйста, полностью заполните форму.")
                    return

            # Ensure data is available for clustering and forecasting
            if 'cleaned_df' not in st.session_state or 'doc2vec_vacancy_vector' not in st.session_state:
                st.error("Данные недоступны. Пожалуйста, сначала получите и обработайте вакансии.")
                return

            # Prepare data for clustering
            doc2vec_dataset_vectors: np.ndarray = np.array(st.session_state.cleaned_df['doc2vec_vector'].tolist())
            logging.info(f"Shape of cleaned dataset: {st.session_state.cleaned_df.shape}")
            logging.info(f"Shape of Doc2Vec vectors: {doc2vec_dataset_vectors.shape}")

            combined_vectors_doc2vec: np.ndarray = np.vstack([doc2vec_dataset_vectors, st.session_state.doc2vec_vacancy_vector])
            umap_params: Dict[str, int] = {'n_neighbors': 10, 'n_components': 2, 'min_dist': 0.3, 'random_state': 42}
            optics_param_grid: Dict[str, List[int]] = {'min_samples': [3, 5, 10], 'xi': [0.05, 0.1, 0.2]}
            clusterer_doc2vec = OPTICS_UMAP_Clusterer(umap_params, optics_param_grid)
            clusterer_doc2vec.fit(combined_vectors_doc2vec)

            # Assign clusters to the dataset and the new vacancy
            st.session_state.cleaned_df['cluster_OPTICS_Doc2Vec'] = clusterer_doc2vec.clusters[:-1]
            job_vector_cluster_doc2vec: int = clusterer_doc2vec.clusters[-1]
            logging.info(f"Cluster of the new job vacancy vector (Doc2Vec): {job_vector_cluster_doc2vec}")

            # Calculate salary statistics for the cluster
            low_salary_stats_doc2vec: Dict[str, float] = get_salary_stats(job_vector_cluster_doc2vec, st.session_state.cleaned_df, 'Low salary')
            high_salary_stats_doc2vec: Dict[str, float] = get_salary_stats(job_vector_cluster_doc2vec, st.session_state.cleaned_df, 'High salary')

            logging.info(f"Low Salary Stats (Doc2Vec): {low_salary_stats_doc2vec}")
            logging.info(f"High Salary Stats (Doc2Vec): {high_salary_stats_doc2vec}")

            user_salary = st.session_state.vacancy_details['vacancy_salary']
            user_salary = float(user_salary) if user_salary else 0

            # Вычисление ошибки (интервала)
            # low_error = user_salary - low_salary_stats_doc2vec['median'] if user_salary > low_salary_stats_doc2vec['median'] else low_salary_stats_doc2vec['median'] - user_salary
            # high_error = high_salary_stats_doc2vec['median'] - user_salary if user_salary < high_salary_stats_doc2vec['median'] else user_salary - high_salary_stats_doc2vec['median']

            # Проверка наличия значений медианы
            low_median = low_salary_stats_doc2vec.get('median', None)
            high_median = high_salary_stats_doc2vec.get('median', None)

            st.header("Результаты")
            with st.expander("Показать/Скрыть результаты", expanded=True):
                # Стиль для графиков
                st.markdown("""
                    <style>
                    .plot-container {
                        display: flex;
                        flex-direction: row;
                        justify-content: space-between;
                    }
                    .plot-item {
                        flex: 1;
                        margin: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                
                # Объединенный график для низкой и высокой зарплаты
                combined_salary_fig = go.Figure()

                # Добавление линии, соединяющей медианные значения низкой и высокой зарплат
                if not np.isnan(low_median) and not np.isnan(high_median) and low_median < high_median:
                    combined_salary_fig.add_trace(go.Scatter(
                        x=[low_median, high_median],
                        y=['User Salary', 'User Salary'],
                        mode='lines+markers',
                        line=dict(color='blue'),
                        marker=dict(symbol='line-ew', size=10, color='blue'),
                        name='Median Salary Interval'
                    ))
                elif not np.isnan(low_median) and not np.isnan(high_median): 
                    combined_salary_fig.add_trace(go.Scatter(
                        x=[high_median, low_median],
                        y=['User Salary', 'User Salary'],
                        mode='lines+markers',
                        line=dict(color='blue'),
                        marker=dict(symbol='line-ew', size=10, color='blue'),
                        name='Median Salary Interval'
                    ))
                elif np.isnan(low_median) and not np.isnan(high_median):
                    combined_salary_fig.add_trace(go.Scatter(
                        x=[0, high_median],
                        y=['User Salary', 'User Salary'],
                        mode='lines+markers',
                        line=dict(color='blue'),
                        marker=dict(symbol='line-ew', size=10, color='blue'),
                        name='Median Salary Interval'
                    ))
                elif not np.isnan(low_median) and np.isnan(high_median):
                    combined_salary_fig.add_trace(go.Scatter(
                        x=[low_median, 300000],
                        y=['User Salary', 'User Salary'],
                        mode='lines+markers',
                        line=dict(color='blue'),
                        marker=dict(symbol='line-ew', size=10, color='blue'),
                        name='Median Salary Interval'
                    ))
                elif low_median == high_median:
                    combined_salary_fig.add_trace(go.Scatter(
                        x=[low_median],
                        y=['User Salary'],
                        mode='markers+text',
                        marker=dict(color='blue', size=12, symbol="circle"),
                        text=["Median Salary"],
                        textposition="top center",
                        name="Median Salary"
                    ))


                # Добавление пользовательской зарплаты в виде точки
                combined_salary_fig.add_trace(go.Scatter(
                    x=[user_salary],
                    y=['User Salary'],
                    mode='markers+text',
                    marker=dict(color='red', size=12, symbol="circle"),
                    text=["User Salary"],
                    textposition="top center",
                    name="User Salary"
                ))

                combined_salary_fig.update_layout(
                    title="Зарплата пользователя и интервал медианной зарплаты",
                    xaxis_title="Зарплата",
                    yaxis_title="Метрики",
                    template="simple_white"
                )

                st.plotly_chart(combined_salary_fig, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

