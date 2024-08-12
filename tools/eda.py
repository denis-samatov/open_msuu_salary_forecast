# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# from wordcloud import WordCloud
# import streamlit as st




# def visualize_salary_distribution(df):
    # fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    # sns.histplot(df[df['Low salary'] > 0]['Low salary'], kde=True, ax=axes[0], color='blue', bins=10)
    # axes[0].set_title('Low Salary Distribution')
    # axes[0].set_xlabel('Low Salary')
    
    # sns.histplot(df[df['High salary'] > 0]['High salary'], kde=True, ax=axes[1], color='orange', bins=10)
    # axes[1].set_title('High Salary Distribution')
    # axes[1].set_xlabel('High Salary')
    
    # plt.suptitle('Salary Distribution', fontsize=16)
    # plt.tight_layout()
    # st.pyplot(fig)
    # plt.close(fig)

# def visualize_salary_boxplots(df):
#     df_long = pd.melt(df, value_vars=['Low salary', 'High salary'], var_name='Salary Type', value_name='Salary')
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 7))
#     sns.boxplot(x='Salary Type', y='Salary', data=df_long[df_long['Salary'] > 0], ax=axes[0], palette=['blue', 'orange'])
#     axes[0].set_title('Salary Boxplot')
    
#     sns.swarmplot(x='Salary Type', y='Salary', data=df_long[df_long['Salary'] > 0], ax=axes[1], palette=['blue', 'orange'])
#     axes[1].set_title('Salary Swarmplot')
    
#     plt.suptitle('Salary Analysis', fontsize=16)
#     plt.tight_layout()
#     st.pyplot(fig)

# def visualize_key_skills(df):
    # all_skills = ",".join(df['Key skills'].dropna()).split(",")
    # all_skills = [skill.strip() for skill in all_skills if skill.strip()]
    # skill_counts = Counter(all_skills)
    
    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_counts)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.imshow(wordcloud, interpolation='bilinear')
    # ax.axis('off')
    # st.pyplot(fig)
    
#     most_common_words = skill_counts.most_common(10)
#     words, frequencies = zip(*most_common_words)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.barplot(x=list(frequencies), y=list(words), ax=ax)
#     ax.set_xlabel('Frequency')
#     ax.set_ylabel('Skills')
#     ax.set_title('Top 10 Key Skills')
#     st.pyplot(fig)

# def visualize_professional_roles(df):
#     all_roles = df['Professional roles']
#     all_roles = [role.strip() for role in all_roles]
#     role_counts = Counter(all_roles)
    
#     most_common_roles = role_counts.most_common(10)
#     words, frequencies = zip(*most_common_roles)
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.barplot(x=list(frequencies), y=list(words), ax=ax)
#     ax.set_xlabel('Frequency')
#     ax.set_ylabel('Roles')
#     ax.set_title('Top 10 Professional Roles')
#     st.pyplot(fig)
    