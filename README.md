# Salary Forecasting

## Table of Contents
- [Salary Forecasting](#salary-forecasting)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [File Descriptions](#file-descriptions)

## Overview
This application leverages job posting data from HeadHunter (HH) to predict potential salaries for various positions. Below is a brief overview of the process:

1. **Data Collection:** The app fetches job posting data from HeadHunter (HH) using API access.
2. **Select Options:** Users can select specific regions they are interested in, select a job title, and select a professional role to see local job posting data.
3. **Data Processing:** The collected data is subject to final cleaning and pre-processing to extract key information. In addition, vacancies are vectorized.
4. **Analysis:** The application performs several analyses including anomaly detection, salary distribution, key skills and job role analysis.
5. **Forecasting:** Uses advanced machine learning algorithms to predict possible salaries for specified positions, providing  historical data.

## Installation
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
1. **Login:** Use `login.py` to authenticate.
2. **Data Collection:** Run `parser.py` and `process_vacancies.py` to collect and process job vacancies data.
3. **Data Analysis:** Utilize `eda.py` for exploratory data analysis and `anomaly_detection.py` for detecting anomalies in the data.
4. **Forecasting:** Use `salary_forecasting.py` to forecast salaries based on the processed data.
5. **Visualization:** Generate charts using `charts.py`.

## File Descriptions
- `anomaly_detection.py`: Scripts for identifying anomalies in the dataset.
- `api_links_and_constant.py`: Contains API links and constant values used throughout the project.
- `charts.py`: Functions to create visualizations and charts.
- `create_dataset.py`: Scripts to create datasets from raw data.
- `eda.py`: Scripts for exploratory data analysis.
- `get_hh_key.py`: Script to obtain API keys from HeadHunter.
- `parser.py`: Main parser script to extract job vacancies data.
- `process_vacancies.py`: Processes the raw vacancies data into a structured format.
- `salary_forecasting.py`: Scripts for forecasting salaries using machine learning models.
- `login.py`: Script to handle user authentication.
- `logout.py`: Script to handle user logout.
- `main.py`: Main script to run the application.
- `requirements.txt`: File containing the list of required Python packages.
- `show_forecast_salary.py`: Script to display forecasted salaries.
- `show_program_description.py`: Script to display the program's description.
