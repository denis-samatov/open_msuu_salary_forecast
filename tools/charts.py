import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go


from plotly.subplots import make_subplots
from collections import Counter
from wordcloud import WordCloud




def plot_salary_distribution(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Plot salary distribution using histograms for low and high salaries.

    Parameters:
    - cleaned_df (pd.DataFrame): DataFrame containing cleaned salary data.

    Returns:
    - fig (go.Figure): Plotly figure containing the salary distribution histograms.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Распределение низких зарплат", "Распределение высоких зарплат"))

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
    
    # Create the bar chart with adjustments for better text rendering
    fig = px.bar(x=list(frequencies), y=list(words), orientation='h', 
                 labels={'x':'Частота', 'y':'Навыки'}, title='Топ-10 ключевых навыков',
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Adjust layout to prevent text overlap
    fig.update_layout(height=500,  # Increase height for better readability
                      margin=dict(l=300, r=50, t=50, b=50),  # Add more margin on the left for long labels
                      yaxis_tickangle=0,  # Set angle of Y-axis labels
                      autosize=False,  # Disable autosize to manually control dimensions
                      width=800  # Set fixed width for the figure
                     )
    
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
                 title='Boxplot зарплат', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(height=400)  # Set fixed height here

    
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
                labels={'x':'Частота', 'y':'Роли'}, title='Топ-10 профессиональных ролей',
                color_discrete_sequence=px.colors.qualitative.Plotly)
    
    fig.update_layout(height=500,  # Increase height for better readability
                margin=dict(l=200, r=50, t=50, b=50),  # Add more margin on the left for long labels
                yaxis_tickangle=0,  # Set angle of Y-axis labels
                autosize=False,  # Disable autosize to manually control dimensions
                width=800  # Set fixed width for the figure
                )  # Set fixed height here
    return fig

def add_median_interval_trace(fig, low_median, high_median):
    """
    Adds median salary interval or point to the figure based on the conditions.

    Args:
    - fig: Plotly figure object.
    - low_median: The low median salary.
    - high_median: The high median salary.

    Returns:
    - None: Modifies the figure in place.
    """
    if not np.isnan(low_median) and not np.isnan(high_median):
        if low_median < high_median:
            x_vals = [low_median, high_median]
        else:
            x_vals = [high_median, low_median]
    elif np.isnan(low_median) and not np.isnan(high_median):
        x_vals = [0, high_median]
    elif not np.isnan(low_median) and np.isnan(high_median):
        x_vals = [low_median, 300000]
    else:
        x_vals = None

    # Add the median interval line if applicable
    if x_vals:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=['Зарплата пользователя', 'Зарплата пользователя'],
            mode='lines+markers',
            line=dict(color='blue'),
            marker=dict(symbol='line-ew', size=10, color='blue'),
            name='Интервал медианной зарплаты'
        ))

    # If low_median == high_median or one of them is NaN, mark the point
    if low_median == high_median or x_vals is None:
        median_point = low_median if not np.isnan(low_median) else high_median
        fig.add_trace(go.Scatter(
            x=[median_point],
            y=['Зарплата пользователя'],
            mode='markers+text',
            marker=dict(color='blue', size=12, symbol="circle"),
            text=["Медианная зарплата"],
            textposition="top center",
            name="Медианная зарплата"
        ))

def plot_combined_salary(combined_salary_fig, low_median, high_median, user_salary):
    """
    Plots the combined salary figure with user salary and median salary intervals.

    Args:
    - combined_salary_fig: Plotly figure object.
    - low_median: The low median salary.
    - high_median: The high median salary.
    - user_salary: The user's salary.

    Returns:
    - None: Modifies the figure in place.
    """
    # Add median salary interval trace
    add_median_interval_trace(combined_salary_fig, low_median, high_median)

    # Add user salary point
    combined_salary_fig.add_trace(go.Scatter(
        x=[user_salary],
        y=['Зарплата пользователя'],
        mode='markers+text',
        marker=dict(color='red', size=12, symbol="circle"),
        text=["Зарплата пользователя"],
        textposition="top center",
        name="Зарплата пользователя"
    ))

    # Update layout
    combined_salary_fig.update_layout(
        title="Зарплата пользователя и интервал медианной зарплаты",
        xaxis_title="Зарплата",
        yaxis_title="Метрики",
        template="simple_white"
    )

    return combined_salary_fig
