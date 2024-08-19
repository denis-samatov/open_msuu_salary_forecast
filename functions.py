import numpy as np
import requests
import logging
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import os
import nltk
import re
import string
import random
import math
import time

from plotly.subplots import make_subplots
from collections import Counter
from wordcloud import WordCloud
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from umap import UMAP
from gensim.models.doc2vec import Doc2Vec
from typing import Any, Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.autonotebook import tqdm


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    data = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_page = {executor.submit(fetch_vacancies_data, page, search_params): page for page in range(pages)}
        for future in tqdm(as_completed(future_to_page), total=pages, desc="Pages"):
            page = future_to_page[future]
            try:
                vacancies = future.result()
                logging.info(f"Page {page}: {len(vacancies)} vacancies found.")

                vacancy_ids = [vacancy.get('id') for vacancy in vacancies]
                unique_vacancy_ids = set(vacancy_ids)
                logging.info(f'Number of unique vacancies: {len(unique_vacancy_ids)}')

                future_to_vacancy = {executor.submit(extract_vacancy_info, vacancy_id): vacancy_id for vacancy_id in unique_vacancy_ids}
                for num, future_vacancy in enumerate(tqdm(as_completed(future_to_vacancy), total=len(unique_vacancy_ids), desc="Vacancies")):
                    vacancy_info = future_vacancy.result()
                    if vacancy_info:
                        data.append(vacancy_info)
            except Exception as e:
                logging.error(f'Error processing page {page}: {e}')

    df_vacancies = pd.DataFrame(data)
    df_vacancies[['Low salary', 'High salary', 'Average salary']] = df_vacancies.apply(convert_and_average_salary, axis=1)

    return df_vacancies

# Загрузка необходимых данных NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def download_nltk_data(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], download_dir=nltk_data_dir)

download_nltk_data('tokenizers/punkt')
download_nltk_data('corpora/stopwords')

# Set random seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

# Initialize stop words and lemmatizers
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
lemmatizer_en = WordNetLemmatizer()
morph_ru = MorphAnalyzer()

# Compile regex patterns
url_pattern = re.compile(r'http[s]?://\S+')
email_pattern = re.compile(r'\S+@\S+')
phone_pattern = re.compile(r'\b\d{1,3}[-.\s]??\d{1,3}[-.\s]??\d{1,3}[-.\s]??\d{1,4}\b')
date_pattern = re.compile(r'\b\d{1,3}[-.\s]??\d{2}[-.\s]??\d{2}\b')
social_media_pattern = re.compile(r'\b(?:telegram|почта|телеграм|whatsapp|ватсап|телег|tg|вконтакте|vk|github|git)\s\S+', flags=re.IGNORECASE)
social_media_link_pattern = re.compile(r'\b(?:vk\.com|github\.com)/\S+', flags=re.IGNORECASE)
special_char_pattern = re.compile(r"\\n\\n|\\n\\|\\n|\n-|•|●|»|«|\\r|\\r r|``|''|“|”|·|``|_")
emoji_pattern = re.compile(r"⚡️|✅|✔|❌|⛔|❤️|1️⃣|2️⃣|3️⃣|4️⃣|5️⃣|6️⃣|7️⃣|❗")
html_pattern = re.compile(r'<[^>]+>')

def preprocess_text(text: str) -> str:
    """
    Preprocess text by normalizing, removing unwanted characters, and lemmatizing.

    Args:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    text = text.lower()
    text = url_pattern.sub('', text)
    text = email_pattern.sub('', text)
    text = phone_pattern.sub('', text)
    text = date_pattern.sub('', text)
    text = social_media_pattern.sub('', text)
    text = social_media_link_pattern.sub('', text)
    text = special_char_pattern.sub('', text)
    text = emoji_pattern.sub('', text)
    text = html_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('..', '.')

    words = word_tokenize(text)
    processed_words = []
    for word in words:
        if word not in string.punctuation:
            if word not in stop_words_en and word not in stop_words_ru:
                lemmatized_word = lemmatizer_en.lemmatize(word)
                lemmatized_word_ru = morph_ru.normal_forms(lemmatized_word)[0]
                processed_words.append(lemmatized_word_ru)
        else:
            processed_words.append(word)

    return ' '.join(processed_words)

def process_texts(texts: List[str]) -> List[str]:
    """
    Preprocess a list of texts using joblib for parallel processing.

    Args:
    texts (List[str]): A list of texts to preprocess.

    Returns:
    List[str]: A list of preprocessed texts.
    """

    return Parallel(n_jobs=-1)(delayed(preprocess_text)(text) for text in texts)

def drop_punctuation_text(text: str) -> str:
    """
    Remove punctuation and lemmatize text.

    Args:
    text (str): The input text.

    Returns:
    str: The text without punctuation and lemmatized.
    """
    words = word_tokenize(text)
    processed_words = [morph_ru.normal_forms(lemmatizer_en.lemmatize(word))[0]
                       for word in words if word not in stop_words_en and word not in stop_words_ru and word not in string.punctuation]
    return ' '.join(processed_words)

def create_combined_text(row: pd.Series) -> str:
    """
    Combine multiple fields into a single text field for processing.

    Args:
    row (pd.Series): A row of a DataFrame containing the fields.

    Returns:
    str: The combined text.
    """
    
    return (
        f"Название вакансии: {safe_value(row['Title'])}. "
        f"Профессиональная роль: {safe_value(row['Professional roles'])}. "
        f"Описание вакансии: {safe_value(row['Description'])}. "
        f"Навыки: {safe_value(row['Key skills'])}. "
        f"Опыт работы: {safe_value(row['Experience'])}. "
        f"Тип занятости: {safe_value(row['Employment'])}. "
        f"График работы: {safe_value(row['Schedule'])}."
    )

def create_combined_texts(df: pd.DataFrame) -> pd.Series:
    """
    Apply the creation of combined text for each row in a DataFrame in parallel.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.Series: A Series of combined texts.
    """
    return Parallel(n_jobs=-1)(delayed(create_combined_text)(row) for _, row in df.iterrows())

def create_dataset(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, Doc2Vec]:
    """
    Preprocess dataset and generate embeddings.

    Args:
    dataset (pd.DataFrame): The input dataset to process.

    Returns:
    Tuple[pd.DataFrame, Doc2Vec]: The processed dataset and the Doc2Vec model.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset = dataset.dropna(subset=['Description'])

    dataset['combined_text'] = create_combined_texts(dataset)
    dataset['processed_combined_text'] = process_texts(dataset['combined_text'])

    doc2vec_text = dataset['processed_combined_text'].apply(drop_punctuation_text)
    tagged_documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(doc2vec_text)]
    doc2vec_model = Doc2Vec(vector_size=200, window=5, min_count=3, workers=4, epochs=40)
    doc2vec_model.random.seed(SEED)
    doc2vec_model.build_vocab(tagged_documents)
    doc2vec_model.train(tagged_documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    dataset['doc2vec_vector'] = doc2vec_text.apply(lambda text: doc2vec_model.infer_vector(text.split()))

    return dataset, doc2vec_model

def plot_salary_distribution(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Plot salary distribution using histograms for low and high salaries.

    Parameters:
    - cleaned_df (pd.DataFrame): DataFrame containing cleaned salary data.

    Returns:
    - fig (go.Figure): Plotly figure containing the salary distribution histograms.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Low Salary Distribution", "High Salary Distribution"))

    fig.add_trace(
        go.Histogram(
            x=cleaned_df[cleaned_df['Low salary'] > 0]['Low salary'],
            nbinsx=10,
            marker_color='blue',
            name='Низкая зарплата'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Histogram(
            x=cleaned_df[cleaned_df['High salary'] > 0]['High salary'],
            nbinsx=10,
            marker_color='orange',
            name='Высокая зарплата'
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title_text='Распределение зарплат',
        bargap=0.1,
        showlegend=False,
        height=400,
        width=700
    )

    fig.update_xaxes(title_text="Низкая зарплата", row=1, col=1)
    fig.update_xaxes(title_text="Высокая зарплата", row=1, col=2)
    fig.update_yaxes(title_text="Количество", row=1, col=1)

    return fig

def plot_key_skills(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Plot the top 10 key skills using a bar chart.

    Parameters:
    - cleaned_df (pd.DataFrame): DataFrame containing cleaned key skills data.

    Returns:
    - fig (px.Figure): Plotly figure containing the bar chart of top 10 key skills.
    """

    all_skills = ",".join(cleaned_df['Key skills'].dropna()).split(",")
    all_skills = [skill.strip() for skill in all_skills if skill.strip()]
    skill_counts = Counter(all_skills)

    most_common_words = skill_counts.most_common(10)
    words, frequencies = zip(*most_common_words)
    fig = px.bar(x=list(frequencies), y=list(words), orientation='h', 
                 labels={'x':'Частота', 'y':'Навыки'}, title='Топ-10 ключевых навыков')
    fig.update_layout(height=400)
    return fig

def plot_key_skills_wordcloud(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Generate a word cloud of key skills and plot it using Plotly.

    Parameters:
    - cleaned_df (pd.DataFrame): DataFrame containing cleaned key skills data.

    Returns:
    - fig (go.Figure): Plotly figure containing the word cloud.
    """
    all_skills = ",".join(cleaned_df['Key skills'].dropna()).split(",")
    all_skills = [skill.strip() for skill in all_skills if skill.strip()]
    skill_counts = Counter(all_skills)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_counts)
    wordcloud_image = wordcloud.to_array()

    fig = go.Figure()
    fig.add_trace(go.Image(z=wordcloud_image))
    fig.update_layout(
        title='Облако слов ключевых навыков',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=400
    )
    return fig

def plot_salary_boxplots(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Plot salary boxplots for low and high salaries.

    Parameters:
    - cleaned_df (pd.DataFrame): DataFrame containing cleaned salary data.

    Returns:
    - fig (px.Figure): Plotly figure containing the salary boxplots.
    """

    df_long = pd.melt(cleaned_df, value_vars=['Low salary', 'High salary'], 
                      var_name='Тип зарплаты', value_name='Зарплата')

    fig = px.box(df_long[df_long['Зарплата'] > 0], x='Тип зарплаты', y='Зарплата', color='Тип зарплаты', 
                 title='Boxplot зарплат')
    fig.update_layout(height=400)
    return fig

def plot_professional_roles(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Plot the top 10 professional roles using a bar chart.

    Parameters:
    - cleaned_df (pd.DataFrame): DataFrame containing cleaned professional roles data.

    Returns:
    - fig (px.Figure): Plotly figure containing the bar chart of top 10 professional roles.
    """
    all_roles = cleaned_df['Professional roles']
    all_roles = [role.strip() for role in all_roles]
    role_counts = Counter(all_roles)

    most_common_roles = role_counts.most_common(10)
    words, frequencies = zip(*most_common_roles)
    fig = px.bar(x=list(frequencies), y=list(words), orientation='h', 
                 labels={'x':'Частота', 'y':'Роли'}, title='Топ-10 профессиональных ролей')
    fig.update_layout(height=400)
    return fig

def convert_data_to_train(data_string: str) -> List[float]:
    """
    Convert a string of data into a list of floats for training.
    
    Parameters:
    - data_string (str): String containing data values separated by whitespace
    
    Returns:
    - List[float]: List of floats extracted from the data string
    """
    data_string = data_string.strip()[1:-1]
    data_list = re.split(r'\s+', data_string)
    data_list = [float(x.rstrip(',')) for x in data_list if x]
    return data_list

def vectorize_vacancy(vacancy_title: str, vacancy_details: str, doc2vec_model: Doc2Vec) -> np.ndarray:
    """
    Vectorize the given vacancy details using a pre-trained Doc2Vec model.

    Parameters:
    - vacancy_title (str): Title of the vacancy
    - vacancy_details (str): Detailed description of the vacancy
    - doc2vec_model (Doc2Vec): Pre-trained Doc2Vec model
    
    Returns:
    - np.ndarray: Vector representation of the vacancy details
    """
    preprocessed_vacancy = preprocess_text(f"Название: {vacancy_title}. {vacancy_details}")
    preprocessed_vacancy_doc2vec = drop_punctuation_text(preprocessed_vacancy)
    doc2vec_model.random.seed(SEED)
    doc2vec_vector = doc2vec_model.infer_vector(preprocessed_vacancy_doc2vec.split()).reshape(1, -1)

    return doc2vec_vector

class OPTICS_UMAP_Clusterer:
    def __init__(self, umap_params: Dict[str, Any], optics_param_grid: Dict[str, List[Any]]) -> None:
        """
        Initialize the OPTICS_UMAP_Clusterer with UMAP parameters and a grid of OPTICS parameters.
        
        Parameters:
        - umap_params (Dict[str, Any]): Parameters for UMAP.
        - optics_param_grid (Dict[str, List[Any]]): Grid of parameters for OPTICS.
        """

        self.umap_params = umap_params
        self.umap_params['random_state'] = SEED
        self.optics_param_grid = optics_param_grid
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf
        self.clusters: Optional[np.ndarray] = None
        self.optics_model: Optional[OPTICS] = None
        self.umap_model: Optional[UMAP] = None

    def fit(self, data: np.ndarray) -> 'OPTICS_UMAP_Clusterer':
        """
        Fit the UMAP and OPTICS models to the data and find the best clustering parameters.
        
        Parameters:
        - data (np.ndarray): The data to fit the models to.
        
        Returns:
        - self: The fitted OPTICS_UMAP_Clusterer instance.
        """
        self.umap_model = UMAP(**self.umap_params)
        X_umap = self.umap_model.fit_transform(data)
        logging.info("UMAP transformation completed.")

        for params in ParameterGrid(self.optics_param_grid):
            optics = OPTICS(min_samples=params['min_samples'], xi=params['xi'])
            clusters = optics.fit_predict(X_umap)

            n_labels = len(set(clusters)) - (1 if -1 in clusters else 0)
            if n_labels > 1:
                score = silhouette_score(X_umap, clusters)
                logging.info(f"Params: {params}, Silhouette Score: {score}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params

        if self.best_params is not None:
            self.optics_model = OPTICS(min_samples=self.best_params['min_samples'], xi=self.best_params['xi'])
            self.clusters = self.optics_model.fit_predict(X_umap)
            logging.info(f"Best OPTICS params: {self.best_params}")
        else:
            raise ValueError("No valid hyperparameters found for OPTICS clustering.")

        return self

    def plot_clusters(self) -> None:
        """
        Plot the clusters found by the UMAP and OPTICS models.
        """

        if self.clusters is not None and self.umap_model is not None:
            plt.scatter(self.umap_model.embedding_[:, 0], self.umap_model.embedding_[:, 1], c=self.clusters, cmap='Spectral', s=5)
            plt.colorbar()
            plt.title("UMAP projection of the clusters")
            plt.show()
        else:
            raise ValueError("UMAP or clustering results are not available for plotting.")

def get_salary_stats(cluster: int, df: pd.DataFrame, salary_col: str) -> Dict[str, float]:
    """
    Calculate statistical measures for the salaries of a given cluster.
    
    Parameters:
    - cluster (int): The cluster number.
    - df (pd.DataFrame): The DataFrame containing the data.
    - salary_col (str): The column name of the salary data.
    
    Returns:
    - Dict[str, float]: A dictionary with statistical measures of the salaries.
    """
    valid_salaries = df[df['cluster_OPTICS_Doc2Vec'] == cluster][salary_col]
    valid_salaries = valid_salaries[valid_salaries != 0]
    return {
        'median': valid_salaries.median(),
        'mean': valid_salaries.mean(),
        'min': valid_salaries.min(),
        'max': valid_salaries.max(),
        'std': valid_salaries.std()
    }

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect anomalies in the salary columns using Isolation Forest.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Low salary' and 'High salary' columns.
    - contamination (float): The proportion of outliers in the data set.

    Returns:
    - cleaned_df (pd.DataFrame): DataFrame without anomalies.
    - anomalies_df (pd.DataFrame): DataFrame containing only the anomalies.
    """
    required_columns = ['Low salary', 'High salary']

    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"DataFrame must contain '{col}' column")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"'{col}' column must contain numeric data")

    isolation_forest_model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = isolation_forest_model.fit_predict(df[required_columns])

    cleaned_df = df[df['anomaly'] == 1].drop(columns=['anomaly'])
    anomalies_df = df[df['anomaly'] == -1].drop(columns=['anomaly'])

    return cleaned_df, anomalies_df

def get_areas() -> Dict[str, int]:
    """
    Fetch areas from the API and return a dictionary of area names and their IDs.
    
    Returns:
        Dict[str, int]: A dictionary mapping area names to their IDs.
    """
    response = requests.get(API_HH_AREAS)
    response.raise_for_status()
    areas_data = response.json()
    areas = {area['name']: area['id'] for country in areas_data for area in country['areas']}
    return areas

def get_professional_roles() -> Dict[str, int]:
    """
    Fetch professional roles from the API and return a dictionary of role names and their IDs.
    
    Returns:
        Dict[str, int]: A dictionary mapping professional role names to their IDs.
    """
    response = requests.get(API_HH_PROFESSIONAL_ROLES)
    response.raise_for_status()
    roles_data = response.json()
    roles = {role['name']: role['id'] for category in roles_data['categories'] for role in category['roles']}
    return roles

def fetch_vacancies_cached(search_params: Dict[str, List], pages: int) -> pd.DataFrame:
    """
    Fetch vacancies based on search parameters and number of pages.

    Args:
        search_params (Dict[str, List]): The search parameters for fetching vacancies.
        pages (int): The number of pages to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched vacancies.
    """
    return fetch_and_process_vacancies(search_params, pages=pages)

def process_dataset(df_vacancies: pd.DataFrame) -> Tuple[pd.DataFrame, Doc2Vec]:
    return create_dataset(df_vacancies)

def detect_anomalies_cached(df_vacancies_hh: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect anomalies in the dataset and separate them from the cleaned data.

    Args:
        df_vacancies_hh (pd.DataFrame): The DataFrame containing processed vacancy data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the cleaned DataFrame and a DataFrame of anomalies.
    """
    return detect_anomalies(df_vacancies_hh)

def validate_salary(salary: str) -> bool:
    """
    Validate if the input salary is a valid integer.

    Args:
        salary (str): The input salary as a string.

    Returns:
        bool: True if the salary is a valid integer, False otherwise.
    """
    try:
        int(salary)
        return True
    except ValueError:
        return False

def show_forecast_salary() -> None:
    """
    Main function to display the salary forecasting application. Handles user inputs,
    fetches vacancies, processes data, and displays analysis and forecasting results.
    """

    print("Прогнозирование зарплат")

    areas = get_areas()
    professional_roles = get_professional_roles()

    selected_areas: List[str] = ["Москва"]
    area_ids: List[int] = [areas[area] for area in selected_areas]

    text: str = "Data Scientist"
    per_page: int = 100
    pages: int = 3
    only_with_salary: bool = True
    exchanges: List[str] = ["RUB", "USD", "EUR"]
    selected_roles: List[str] = ["Data Scientist"]
    professional_roles_ids: List[int] = [professional_roles[role] for role in selected_roles]

    search_params: Dict[str, List] = {
        'area': area_ids,
        'text': text,
        'per_page': per_page,
        'only_with_salary': only_with_salary,
        "exchanges": exchanges,
        "professional_roles": professional_roles_ids
    }

    df_vacancies: pd.DataFrame = fetch_vacancies_cached(search_params, pages=pages)
    print("Вакансии успешно получены.")

    df_vacancies_hh, doc2vec_model = process_dataset(df_vacancies)
    print("Набор данных успешно обработан.")

    cleaned_df, anomalies_df = detect_anomalies_cached(df_vacancies_hh)
    print("Аномалии обнаружены и удалены.")

    print("Анализ данных")
    fig_salary_dist = plot_salary_distribution(cleaned_df)
    fig_key_skills = plot_key_skills(cleaned_df)
    fig_key_skills_wordcloud = plot_key_skills_wordcloud(cleaned_df)
    fig_salary_boxplot = plot_salary_boxplots(cleaned_df)
    fig_prof_roles = plot_professional_roles(cleaned_df)

    fig_salary_dist.show()
    fig_key_skills.show()
    fig_key_skills_wordcloud.show()
    fig_salary_boxplot.show()
    fig_prof_roles.show()

    vacancy_details: Dict[str, str] = {
        'vacancy_title': 'Data Scientist',
        'vacancy_professional_role': 'Data Scientist',
        'vacancy_description': 'Experience in machine learning and data analysis',
        'vacancy_salary': '120000',
        'vacancy_experience': '3 years',
        'vacancy_employment': 'Full-time',
        'vacancy_schedule': 'Flexible'
    }

    if validate_salary(vacancy_details['vacancy_salary']):
        vacancy_details['vacancy_salary'] = int(vacancy_details['vacancy_salary'])
    else:
        print("Введите целое число в поле 'Заработная плата'.")
        return

    vacancy_description: str = f"""
    Название вакансии: {vacancy_details['vacancy_title']}.
    Профессиональная роль: {vacancy_details['vacancy_professional_role']}.
    Описание вакансии и навыки: {vacancy_details['vacancy_description']}
    Опыт работы: {vacancy_details['vacancy_experience']}.
    Тип занятости: {vacancy_details['vacancy_employment']}.
    График работы: {vacancy_details['vacancy_schedule']}.
    """

    if vacancy_details['vacancy_title'] and vacancy_details['vacancy_description']:
        doc2vec_vacancy_vector: np.ndarray = vectorize_vacancy(vacancy_details['vacancy_title'], vacancy_description, doc2vec_model)
        print("Вакансия успешно векторизована.")
    else:
        print("Детали вакансии неполные. Пожалуйста, полностью заполните форму.")
        return

    doc2vec_dataset_vectors: np.ndarray = np.array(cleaned_df['doc2vec_vector'].tolist())
    combined_vectors_doc2vec: np.ndarray = np.vstack([doc2vec_dataset_vectors, doc2vec_vacancy_vector])
    umap_params: Dict[str, int] = {'n_neighbors': 10, 'n_components': 2, 'min_dist': 0.3, 'random_state': 42}
    optics_param_grid: Dict[str, List[int]] = {'min_samples': [3, 5, 10], 'xi': [0.05, 0.1, 0.2]}
    clusterer_doc2vec = OPTICS_UMAP_Clusterer(umap_params, optics_param_grid)
    clusterer_doc2vec.fit(combined_vectors_doc2vec)

    cleaned_df['cluster_OPTICS_Doc2Vec'] = clusterer_doc2vec.clusters[:-1]
    job_vector_cluster_doc2vec: int = clusterer_doc2vec.clusters[-1]

    low_salary_stats_doc2vec: Dict[str, float] = get_salary_stats(job_vector_cluster_doc2vec, cleaned_df, 'Low salary')
    high_salary_stats_doc2vec: Dict[str, float] = get_salary_stats(job_vector_cluster_doc2vec, cleaned_df, 'High salary')

    user_salary: float = float(vacancy_details['vacancy_salary'])

    low_median = low_salary_stats_doc2vec.get('median', None)
    high_median = high_salary_stats_doc2vec.get('median', None)

    combined_salary_fig = go.Figure()

    if not np.isnan(low_median) and not np.isnan(high_median):
        combined_salary_fig.add_trace(go.Scatter(
            x=[low_median, high_median],
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

    combined_salary_fig.show()

if __name__ == "__main__":
    show_forecast_salary()
