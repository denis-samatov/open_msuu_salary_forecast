import logging
import re
import numpy as np
from typing import List

from tools.api_links_and_constant import SEED
from tools.create_dataset import preprocess_text, drop_punctuation_text
from gensim.models.doc2vec import Doc2Vec

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(SEED)

def convert_data_to_train(data_string: str) -> List[float]:
    """
    Convert a string of data into a list of floats for training.
    
    Parameters:
    - data_string (str): String containing data values separated by whitespace
    
    Returns:
    - List[float]: List of floats extracted from the data string
    """
    data_string = data_string.strip()[1:-1]  # Remove brackets
    data_list = re.split(r'\s+', data_string)  # Split by whitespace
    
    return [float(x.rstrip(',')) for x in data_list if x] # Convert to floats

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
    # Preprocess the vacancy title and details
    preprocessed_vacancy = preprocess_text(f"Название: {vacancy_title}. {vacancy_details}")

    # Encode the preprocessed text with Doc2Vec
    preprocessed_vacancy_doc2vec = drop_punctuation_text(preprocessed_vacancy)
    doc2vec_vector = doc2vec_model.infer_vector(preprocessed_vacancy_doc2vec.split())

    return doc2vec_vector.reshape(1, -1)
