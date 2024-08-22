import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Dict, List, Optional

from tools.api_links_and_constant import SEED
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from umap import UMAP

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(SEED)

class OPTICS_UMAP_Clusterer:
    def __init__(self, umap_params: Dict[str, Any], optics_param_grid: Dict[str, List[Any]]) -> None:
        """
        Initialize the OPTICS_UMAP_Clusterer with UMAP parameters and a grid of OPTICS parameters.
        
        Parameters:
        - umap_params (Dict[str, Any]): Parameters for UMAP.
        - optics_param_grid (Dict[str, List[Any]]): Grid of parameters for OPTICS.
        """
        self.umap_params = umap_params
        self.umap_params['random_state'] = SEED  # Ensure UMAP uses the fixed seed
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
        # Apply UMAP
        self.umap_model = UMAP(**self.umap_params)
        X_umap = self.umap_model.fit_transform(data)
        logging.info("UMAP transformation completed.")

        # Grid search for OPTICS parameters
        for params in ParameterGrid(self.optics_param_grid):
            optics = OPTICS(min_samples=params['min_samples'], xi=params['xi'])
            clusters = optics.fit_predict(X_umap)

            # Evaluate clustering
            n_labels = len(set(clusters)) - (1 if -1 in clusters else 0)
            if n_labels > 1:
                score = silhouette_score(X_umap, clusters)
                logging.info(f"Params: {params}, Silhouette Score: {score}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params

        # Fit OPTICS with the best parameters
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