import pandas as pd
import logging
import requests
import math
import time
import streamlit as st
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.autonotebook import tqdm

from tools.api_links_and_constant import USD_TO_RUB, EUR_TO_RUB, API_HH_VACANCIES, API_HH_KEY  # Assuming API_HH_KEY is defined for the API key

# Function Definitions
def convert_and_average_salary(row: pd.Series) -> pd.Series:
    """
    Converts salaries to RUB and calculates the average salary.
    
    Args:
    row (pd.Series): A row of a DataFrame containing salary information.
    
    Returns:
    pd.Series: A series with low salary, high salary, and average salary in RUB.
    """
    low_salary = float(row['Low salary']) if row['Low salary'] else 0
    high_salary = float(row['High salary']) if row['High salary'] else 0
    average_salary = (low_salary + high_salary) / 2 if low_salary and high_salary else low_salary or high_salary

    if row['Currency'] == 'USD':
        low_salary *= USD_TO_RUB
        high_salary *= USD_TO_RUB
    elif row['Currency'] == 'EUR':
        low_salary *= EUR_TO_RUB
        high_salary *= EUR_TO_RUB

    return pd.Series([low_salary, high_salary, average_salary], index=['Low salary', 'High salary', 'Average salary'])

def safe_value(value: Any) -> str:
    """
    Safely extract values, returning an empty string if the value is None or NaN.
    
    Args:
    value (Any): The input value.
    
    Returns:
    str: The processed string, empty if the input is None or NaN.
    """
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else value

def fetch_vacancies_data(page: int, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetches vacancy data for a given page with specified search parameters.
    
    Args:
    page (int): The page number to fetch.
    search_params (Dict[str, Any]): The search parameters for the API request.
    
    Returns:
    List[Dict[str, Any]]: A list of vacancy items.
    """
    try:
        headers = {'Authorization': f'Bearer {API_HH_KEY}'}
        response = requests.get(API_HH_VACANCIES, params=search_params, headers=headers)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logging.error(f'Error fetching vacancies data on page {page}: {e}')
        return []

def extract_vacancy_info(vacancy_id: str) -> Dict[str, Any]:
    """
    Extracts detailed information from a vacancy object.
    
    Args:
    vacancy_id (str): The ID of the vacancy to extract.
    
    Returns:
    Dict[str, Any]: A dictionary with detailed vacancy information.
    """
    try:
        headers = {'Authorization': f'Bearer {API_HH_KEY}'}
        response = requests.get(f'{API_HH_VACANCIES}/{vacancy_id}', headers=headers)
        response.raise_for_status()
        vacancy_data = response.json()

        salary = vacancy_data.get("salary", {})
        return {
            'ID': vacancy_id,
            'Title': vacancy_data.get("name", ""),
            'Professional roles': ", ".join(role["name"] for role in vacancy_data.get("professional_roles", [])),
            'Low salary': safe_value(salary.get('from')),
            'High salary': safe_value(salary.get('to')),
            'Currency': safe_value(salary.get('currency')),
            'Employer': vacancy_data.get("employer", {}).get("name", ""),
            'Area': vacancy_data.get("area", {}).get("name", ""),
            'Experience': vacancy_data.get("experience", {}).get("name", ""),
            'Employment': vacancy_data.get("employment", {}).get("name", ""),
            'Schedule': vacancy_data.get("schedule", {}).get("name", ""),
            'Description': vacancy_data.get("description", ""),
            'Key skills': ", ".join(skill["name"] for skill in vacancy_data.get("key_skills", []))
        }
    except requests.exceptions.RequestException as e:
        logging.error(f'Error fetching vacancy data for {vacancy_id}: {e}')
        return {}

def fetch_and_process_vacancies(search_params: Dict[str, Any], pages: int = 3) -> pd.DataFrame:
    """
    Main function to fetch and process vacancies data.
    
    Args:
    search_params (Dict[str, Any]): The search parameters for the API request.
    pages (int): The number of pages to fetch.
    
    Returns:
    pd.DataFrame: A DataFrame with processed vacancies data.
    """
    pages_progress_text = 'Pages'
    vacancies_progress_text = 'Vacancies'
    bar_for_pages = st.progress(0.0, text=pages_progress_text)
    bar_for_vacancies = st.progress(0.0, text=vacancies_progress_text)

    data = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_page = {executor.submit(fetch_vacancies_data, page, search_params): page for page in range(pages)}
        for future in tqdm(as_completed(future_to_page), total=pages, desc="Pages"):
            page = future_to_page[future]
            bar_for_pages.progress((page + 1) / pages, text=f'{pages_progress_text} ({page + 1}/{pages})')
            try:
                vacancies = future.result()
                logging.info(f"Page {page}: {len(vacancies)} vacancies found.")

                vacancy_ids = [vacancy.get('id') for vacancy in vacancies]
                unique_vacancy_ids = set(vacancy_ids)
                logging.info(f'Number of unique vacancies: {len(unique_vacancy_ids)}')

                future_to_vacancy = {executor.submit(extract_vacancy_info, vacancy_id): vacancy_id for vacancy_id in unique_vacancy_ids}
                for num, future_vacancy in enumerate(tqdm(as_completed(future_to_vacancy), total=len(unique_vacancy_ids), desc="Vacancies")):
                    bar_for_vacancies.progress((num + 1) / len(unique_vacancy_ids), text=f'{vacancies_progress_text} ({num + 1}/{len(unique_vacancy_ids)})')
                    vacancy_info = future_vacancy.result()
                    if vacancy_info:
                        data.append(vacancy_info)
            except Exception as e:
                logging.error(f'Error processing page {page}: {e}')
            bar_for_vacancies.empty()
        bar_for_pages.empty()

    # Convert data to pandas DataFrame
    df_vacancies = pd.DataFrame(data)
    df_vacancies[['Low salary', 'High salary', 'Average salary']] = df_vacancies.apply(convert_and_average_salary, axis=1)

    return df_vacancies

# Uncomment the following lines to run the app locally
# if __name__ == "__main__":
#     st.set_page_config(page_title="Vacancy Fetcher", layout="wide")
#     fetch_and_process_vacancies({})
