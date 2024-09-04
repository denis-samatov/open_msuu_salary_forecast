import streamlit as st
import numpy as np
import logging
import plotly.graph_objects as go

from typing import Dict, List
from tools.charts import plot_salary_distribution, plot_key_skills, plot_salary_boxplots, plot_professional_roles, plot_key_skills_wordcloud, plot_combined_salary
from tools.parser import fetch_and_process_vacancies, get_areas, get_professional_roles
from tools.create_dataset import create_dataset
from tools.process_vacancies import vectorize_vacancy
from tools.anomaly_detection import detect_anomalies
from tools.salary_forecasting import OPTICS_UMAP_Clusterer, get_salary_stats
from tools.save_statistics import create_zip_with_stats_and_plots, create_salary_statistics

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

    # with st.spinner("Получение вакансий..."):
    #     df_vacancies = fetch_and_process_vacancies(search_params, pages=pages)
    #     st.session_state.df_vacancies = df_vacancies

    # with st.spinner("Обработка набора данных..."):
    #     df_vacancies_hh, doc2vec_model = create_dataset(st.session_state.df_vacancies)
    #     st.session_state.df_vacancies_hh = df_vacancies_hh
    #     st.session_state.doc2vec_model = doc2vec_model

    # with st.spinner("Обнаружение аномалий..."):
    #     cleaned_df, anomalies_df = detect_anomalies(st.session_state.df_vacancies_hh)
    #     st.session_state.cleaned_df = cleaned_df

    # Display data analysis
    # st.header("Анализ данных")
    # with st.expander("Показать/Скрыть анализ данных", expanded=True):
    #     with st.container():
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             fig_salary_dist = plot_salary_distribution(st.session_state.df_vacancies)
    #             st.plotly_chart(fig_salary_dist, use_container_width=True)

    #             fig_key_skills = plot_key_skills(st.session_state.df_vacancies)
    #             st.plotly_chart(fig_key_skills, use_container_width=True)

    #             fig_key_skills_wordcloud = plot_key_skills_wordcloud(st.session_state.df_vacancies)
    #             st.plotly_chart(fig_key_skills_wordcloud, use_container_width=True)

    #         with col2:
    #             fig_salary_boxplot = plot_salary_boxplots(st.session_state.df_vacancies)
    #             st.plotly_chart(fig_salary_boxplot, use_container_width=True)

    #             fig_prof_roles = plot_professional_roles(st.session_state.df_vacancies)
    #             st.plotly_chart(fig_prof_roles, use_container_width=True)


    # st.header("Добавить вашу вакансию")

    # # Initialize vacancy details if not already in session state
    # st.session_state.vacancy_details = {
    #     'vacancy_title': '',
    #     'vacancy_professional_role': '',
    #     'vacancy_description': '',
    #     'vacancy_salary': '',
    #     'vacancy_experience': '',
    #     'vacancy_employment': '',
    #     'vacancy_schedule': ''
    # }

    # Form for user to input vacancy details
    # with st.expander("Показать/Скрыть форму вакансии", expanded=True):
    #     with st.form(key='vacancy_form'):
    #         st.text_input("Название должности", value=st.session_state.vacancy_details['vacancy_title'], key='vacancy_title', max_chars=100)
    #         st.selectbox(
    #             "Профессиональная роль",
    #             list(professional_roles.keys()),
    #             index=list(professional_roles.keys()).index(st.session_state.vacancy_details['vacancy_professional_role']) if st.session_state.vacancy_details['vacancy_professional_role'] else 0,
    #             key='vacancy_professional_role'
    #         )
    #         st.text_area("Описание работы и навыки", value=st.session_state.vacancy_details['vacancy_description'], key='vacancy_description')
    #         salary_input = st.text_input("Заработная плата", value=st.session_state.vacancy_details['vacancy_salary'], key='vacancy_salary', max_chars=20)

    #         # Check if the input is a valid integer
    #         if validate_salary(salary_input):
    #             st.session_state.vacancy_details['vacancy_salary'] = int(salary_input)
    #         else:
    #             st.error("Введите целое число в поле 'Заработная плата'.")
            
    #         st.text_input("Опыт работы", value=st.session_state.vacancy_details['vacancy_experience'], key='vacancy_experience', max_chars=20)
    #         st.text_input("Тип занятости", value=st.session_state.vacancy_details['vacancy_employment'], key='vacancy_employment', max_chars=30)
    #         st.text_input("График работы", value=st.session_state.vacancy_details['vacancy_schedule'], key='vacancy_schedule', max_chars=20)

    #         submit_button: bool = st.form_submit_button('Векторизовать вакансию и спрогнозировать зарплату')
    #         logging.info(f"Submit button state: {submit_button}")

    # # Process the submitted vacancy form
    # st.session_state.vacancy_details = {
    #     'vacancy_title': st.session_state.vacancy_title,
    #     'vacancy_professional_role': st.session_state.vacancy_professional_role,
    #     'vacancy_description': st.session_state.vacancy_description,
    #     'vacancy_salary': st.session_state.vacancy_salary,
    #     'vacancy_experience': st.session_state.vacancy_experience,
    #     'vacancy_employment': st.session_state.vacancy_employment,
    #     'vacancy_schedule': st.session_state.vacancy_schedule
    # }
    # vacancy_details: str = f"""
    #     Название вакансии: {st.session_state.vacancy_title}.
    #     Профессиональная роль: {st.session_state.vacancy_professional_role}.
    #     Описание вакансии и навыки: {st.session_state.vacancy_description}
    #     Опыт работы: {st.session_state.vacancy_experience}.
    #     Тип занятости: {st.session_state.vacancy_employment}.
    #     График работы: {st.session_state.vacancy_schedule}.
    # """
    # logging.info(f"Vacancy details: {vacancy_details}")

    # # Vectorize the vacancy if the title and description are provided
    # with st.spinner("Векторизация вакансии..."):
    #     doc2vec_vacancy_vector: np.ndarray = vectorize_vacancy(st.session_state.vacancy_details['vacancy_title'], vacancy_details, st.session_state.doc2vec_model)
    #     st.success("Вакансия успешно векторизована.")
    #     st.session_state.doc2vec_vacancy_vector = doc2vec_vacancy_vector

    # # Prepare data for clustering
    # doc2vec_dataset_vectors: np.ndarray = np.array(st.session_state.cleaned_df['doc2vec_vector'].tolist())
    # logging.info(f"Shape of cleaned dataset: {st.session_state.cleaned_df.shape}")
    # logging.info(f"Shape of Doc2Vec vectors: {doc2vec_dataset_vectors.shape}")

    # combined_vectors_doc2vec: np.ndarray = np.vstack([doc2vec_dataset_vectors, st.session_state.doc2vec_vacancy_vector])
    # umap_params: Dict[str, int] = {'n_neighbors': 10, 'n_components': 2, 'min_dist': 0.3, 'random_state': 42}
    # optics_param_grid: Dict[str, List[int]] = {'min_samples': [3, 5, 10], 'xi': [0.05, 0.1, 0.2]}
    # clusterer_doc2vec = OPTICS_UMAP_Clusterer(umap_params, optics_param_grid)
    # clusterer_doc2vec.fit(combined_vectors_doc2vec)

    # # Assign clusters to the dataset and the new vacancy
    # st.session_state.cleaned_df['cluster_OPTICS_Doc2Vec'] = clusterer_doc2vec.clusters[:-1]
    # job_vector_cluster_doc2vec: int = clusterer_doc2vec.clusters[-1]
    # logging.info(f"Cluster of the new job vacancy vector (Doc2Vec): {job_vector_cluster_doc2vec}")

    # # Calculate salary statistics for the cluster
    # low_salary_stats_doc2vec: Dict[str, float] = get_salary_stats(job_vector_cluster_doc2vec, st.session_state.cleaned_df, 'Low salary')
    # high_salary_stats_doc2vec: Dict[str, float] = get_salary_stats(job_vector_cluster_doc2vec, st.session_state.cleaned_df, 'High salary')

    # logging.info(f"Low Salary Stats (Doc2Vec): {low_salary_stats_doc2vec}")
    # logging.info(f"High Salary Stats (Doc2Vec): {high_salary_stats_doc2vec}")

    # user_salary = st.session_state.vacancy_details['vacancy_salary']
    # user_salary = float(user_salary) if user_salary else 0

    # # Проверка наличия значений медианы
    # low_median = low_salary_stats_doc2vec.get('median', None)
    # high_median = high_salary_stats_doc2vec.get('median', None)

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
        
    #     # Объединенный график для низкой и высокой зарплаты
    #     if 'combined_salary_fig' not in st.session_state:
        st.session_state.combined_salary_fig = go.Figure()

        combined_salary_fig = st.session_state.combined_salary_fig
        low_median = 100000
        high_median = 150000

        user_salary = 120000
        combined_salary_fig = plot_combined_salary(combined_salary_fig, low_median, high_median, user_salary)
    
        st.plotly_chart(combined_salary_fig, use_container_width=True)

    #     # Создание сводной таблицы
    #     salary_statisrics_table = create_salary_statistics(low_salary_stats_doc2vec, high_salary_stats_doc2vec) 

    #     other_figures = [fig_salary_dist, fig_key_skills, fig_key_skills_wordcloud, fig_salary_boxplot, fig_prof_roles]
    #     # Генерация ZIP-архива для скачивания
    #     zip_buffer = create_zip_with_stats_and_plots(salary_statisrics_table, combined_salary_fig, other_figures)

    #     # Кнопка для скачивания ZIP-архива
        st.download_button(
            label="Скачать статистику и графики",
            data=combined_salary_fig,
            file_name="salary_data.zip",
            mime="application/zip"
        )

    #     st.markdown('</div>', unsafe_allow_html=True)
