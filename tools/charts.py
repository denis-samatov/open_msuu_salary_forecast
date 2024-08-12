import plotly.express as px
import pandas as pd
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
    fig.update_layout(height=400)  # Set fixed height here
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
                 labels={'x':'Частота', 'y':'Роли'}, title='Топ-10 профессиональных ролей')
    fig.update_layout(height=400)  # Set fixed height here
    return fig