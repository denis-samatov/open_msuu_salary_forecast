import io
import zipfile
import pandas as pd
import plotly.graph_objs as go
from typing import List





def create_salary_statistics(low_salary_stats_doc2vec: dict, high_salary_stats_doc2vec: dict) -> pd.DataFrame:
    """
    Creates a DataFrame with salary statistics for both low and high salaries.

    Args:
    - low_salary_stats_doc2vec (dict): Statistics for low salaries with keys 'median', 'mean', 'min', 'max', 'std'.
    - high_salary_stats_doc2vec (dict): Statistics for high salaries with keys 'median', 'mean', 'min', 'max', 'std'.

    Returns:
    - pd.DataFrame: DataFrame with salary metrics for low and high salaries.
    """
    required_keys = ['median', 'mean', 'min', 'max', 'std']
    
    for key in required_keys:
        if key not in low_salary_stats_doc2vec or key not in high_salary_stats_doc2vec:
            raise ValueError(f"Missing key '{key}' in input dictionaries.")
    
    # Create DataFrame
    salary_stats_df = pd.DataFrame({
        'Salary Type': ['Low Salary'] * 5 + ['High Salary'] * 5,
        'Metric': ['Median', 'Mean', 'Min', 'Max', 'Std'] * 2,
        'Value': [
            low_salary_stats_doc2vec['median'], low_salary_stats_doc2vec['mean'], low_salary_stats_doc2vec['min'],
            low_salary_stats_doc2vec['max'], low_salary_stats_doc2vec['std'],
            high_salary_stats_doc2vec['median'], high_salary_stats_doc2vec['mean'], high_salary_stats_doc2vec['min'],
            high_salary_stats_doc2vec['max'], high_salary_stats_doc2vec['std']
        ]
    })
    
    return salary_stats_df


def create_zip_with_stats_and_plots(salary_stats_df: pd.DataFrame, 
                                    combined_salary_fig: go.Figure, 
                                    other_figures: List[go.Figure]) -> io.BytesIO:
    """
    Creates a ZIP archive containing salary statistics in CSV format and plots as PNGs.
    
    Args:
    - salary_stats_df: DataFrame with salary statistics.
    - combined_salary_fig: Plotly figure with the salary interval chart.
    - other_figures: List of other Plotly figures.
    
    Returns:
    - zip_buffer: Buffer for the ZIP archive to be downloaded.
    """
    zip_buffer = io.BytesIO()
    
    # Ensure figures and data are valid
    if salary_stats_df.empty:
        raise ValueError("The salary statistics DataFrame is empty.")
    
    if combined_salary_fig is None or len(other_figures) == 0:
        raise ValueError("One or more figures are missing.")
    
    # Convert DataFrame to CSV format
    csv_data = salary_stats_df.to_csv(index=False, encoding = "utf-8-sig")
    
    # Create filenames for the figures
    figures = [combined_salary_fig] + other_figures
    filenames = ["salary_chart.png", "salary_distribution.png", "key_skills.png", 
                 "key_skills_wordcloud.png", "salary_boxplot.png", "professional_roles.png"]

    # Create description
    description = create_description()
    
    # Write CSV, description, and figures to ZIP archive
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add CSV file to ZIP
        zip_file.writestr("salary_statistics.csv", csv_data)

        # Add description.txt file to ZIP
        zip_file.writestr("description.txt", description)

        # Save each figure as PNG in the ZIP archive
        for fig, filename in zip(figures, filenames):
            fig_buf = io.BytesIO()
            try:
                fig.write_image(fig_buf, format='png', engine="kaleido")
                fig_buf.seek(0)
                zip_file.writestr(filename, fig_buf.getvalue())
            except Exception as e:
                raise RuntimeError(f"Error saving figure {filename}: {e}")
    
    # Seek to the beginning of the buffer before returning
    zip_buffer.seek(0)


    return zip_buffer


def create_description() -> str:
    """
    Creates a description of the contents of the ZIP file.

    Returns:
    - description (str): The description of the ZIP contents.
    """
    description = """
    Этот архив ZIP содержит следующие файлы:

    1. salary_statistics.csv: CSV-файл, содержащий статистику заработной платы, включая медиану, среднее значение, минимум, максимум и стандартное отклонение как для низких, так и для высоких зарплат.

    2. salary_chart.png: график распределения предсказанной заработной платы.

    3. salary_distribution.png: столбчатая диаграмма распределения заработной платы.

    4. key_skills.png: столбчатая диаграмма, показывающая 10 основных ключевых навыков в наборе данных.

    5. key_skills_wordcloud.png: облако слов, визуализирующее наиболее распространенные ключевые навыки.

    6. salary_boxplot.png: коробчатая диаграмма, визуализирующая распределение заработной платы.

    7. professional_roles.png: столбчатая диаграмма, показывающая 10 основных профессиональных ролей из набора данных.

    Каждый файл изображения представляет собой определенную визуализацию данных о заработной плате и навыках.
    """

    return description
