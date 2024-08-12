import logging
import re
import string
import numpy as np
import random
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm.auto import tqdm
from typing import List, Tuple
from joblib import Parallel, delayed
from tools.api_links_and_constant import SEED

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

def safe_value(value: str) -> str:
    """
    Safely extract values, returning an empty string if the value is None or NaN.
    
    Args:
    value (str): The input string value.

    Returns:
    str: The processed string, empty if the input is None or NaN.
    """
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else value

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

    # Drop rows without descriptions
    dataset = dataset.dropna(subset=['Description'])

    # Combine text fields in parallel
    dataset['combined_text'] = create_combined_texts(dataset)

    # Preprocess combined text in parallel
    dataset['processed_combined_text'] = process_texts(dataset['combined_text'])

    # Generate Doc2Vec embeddings
    doc2vec_text = dataset['processed_combined_text'].apply(drop_punctuation_text)
    tagged_documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(doc2vec_text)]
    doc2vec_model = Doc2Vec(vector_size=200, window=5, min_count=3, workers=4, epochs=40)
    doc2vec_model.random.seed(SEED)
    doc2vec_model.build_vocab(tagged_documents)
    doc2vec_model.train(tagged_documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    dataset['doc2vec_vector'] = doc2vec_text.apply(lambda text: doc2vec_model.infer_vector(text.split()))

    return dataset, doc2vec_model
