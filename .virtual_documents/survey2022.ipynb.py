import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

import plotly.express as px
from ipywidgets import widgets as wg
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

# For show all columns and full value of columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth',300)

# Colors
color_pal = sns.color_palette()
plt.style.use('ggplot')


csvs = glob('survery22_datasets/*20*_responses*.csv')
csvs


csvs_mapping = {
 2018: 'survery22_datasets/2018_kaggle_ds_and_ml_survey_responses_only.csv',
 2019: 'survery22_datasets/2019_kaggle_ds_and_ml_survey_responses_only.csv',
 2020: 'survery22_datasets/2020_kaggle_ds_and_ml_survey_responses_only.csv',
 2021: 'survery22_datasets/kaggle_survey_2021_responses.csv',
 2022: 'survery22_datasets/kaggle_survey_2022_responses.csv'
}


responses = {}
questions = {}
df = []
for year, csv in csvs_mapping.items():
    print(year)
    df_temp = pd.read_csv(csv, low_memory=False).assign(year = year)
    questions[year] = df_temp.loc[0].to_dict() # taking only the questions of every year.
    responses[year] = df_temp.drop(0).reset_index(drop=True) # dropping the questions/(actual questions) by using pd.drop(0) ~ 0 stands for 1st row
    df.append(pd.read_csv(csvs_mapping[year], low_memory=False, header=1).assign(year = year)) # dropping the columns by setting header=1
df = pd.concat(df)


df.head()


fig = plt.figure(figsize=(12, 5))
df['year'].value_counts().sort_index().plot(kind='barh')
plt.show()


year_responses = df['year'].value_counts().sort_index()
year_responses


year_responses[2020] - year_responses[2019]


country = df.groupby('year')['In which country do you currently reside?']


country.value_counts()


post_unstack_country = country.value_counts().unstack()
post_unstack_country


post_unstack_country.sum(axis=0).sort_values(ascending=False)


top_5 = post_unstack_country.sum(axis=0).sort_values(ascending=False).index[:5].to_list()
top_5.append('Bangladesh')
top_5


# post_unstack_country[top_5].plot(kind='bar', figsize=(12, 7))
post_unstack_country[top_5].T.plot(kind='bar', figsize=(12, 7))
plt.xticks(rotation=45)
plt.show()


responses[2022].Q4.value_counts()


responses[2021].Q3.value_counts()


top_5 = post_unstack_country.sum(axis=0).sort_values(ascending=False).index[:6].to_list()
top_5_not_others = [i for i in top_5 if i get_ipython().getoutput("= 'Other']")
top_5_not_others.append('Bangladesh')
top_5_not_others


post_unstack_country[top_5_not_others].plot(kind='bar', figsize=(12, 5), cmap='Blues')
post_unstack_country[top_5_not_others].T.plot(kind='bar', figsize=(12, 5))
plt.xticks(rotation=45, ha='right')
plt.show()


"""
- subset the data (only country exist in top_5_not_others)
- This is the baseline
"""
df_top5 = df.loc[df['In which country do you currently reside?'].isin(top_5_not_others)].reset_index(drop=True).copy().rename(columns={
    'In which country do you currently reside?': 'Country',
    'What is your gender? - Selected Choice': 'Gender',
    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Education',
    'What is your age (# years)?': 'Age'
    
}).fillna(0)



df_top5_g_age = df_top5.groupby(['year', 'Country'])['Age'].value_counts().unstack().fillna(0).astype('int')
df_top5_g_age


name = ['18-21', '25-29', '22-24','30-34', '35-39', '40-44','45-49', '50-54 ', '55-59', '60-69', '70+', '70-79', '80+' ]
value = df_top5[['Age']].value_counts(normalize=True)*100
value


fig = px.pie(
    df_top5, values=value, names=name,
    hole=0.5
)
fig.update_layout()
fig.show()


fig, axs = plt.subplots(2, 1, figsize=(12, 8))
ax = (
    df_top5.query('Country == "United States of America"')\
    .groupby(['Age', 'year']).size()\
    .unstack()\
    .sort_index(ascending=False)\
    .plot(kind='bar',
          stacked=False,
          cmap='Purples',
          ax=axs[0],
          title='Age of Respondents in USA')
)
axs[0].legend(bbox_to_anchor=(1, 1))

ax = (
    df_top5.query('Country == "India"')\
    .groupby(['Age', 'year']).size()\
    .unstack()\
    .sort_index(ascending=False)\
    .plot(kind='bar',
          stacked=False,
          cmap='Greens',
          ax=axs[1],
          title='Age of Respondents in India')
)
axs[1].legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


df_top5_g_age.query('Country == "Bangladesh"')


df_top5['Gender'] = df_top5['Gender'].str.replace('Man', 'Male').str.replace('Woman', 'Female')


name = ['Male','Female', 'Prefer not to say', 'Prefer to self-describe', 'Nonbinary']
value = df_top5[['Gender']].value_counts(normalize=True)*100
value


fig = px.pie(
    df_top5, values=value, names=name,
    hole=0.5
)
fig.update_layout()
fig.show()


df_top5_g_gender_and_country = df_top5.groupby(['year', 'Country'])['Gender'].value_counts(normalize=True)*100
df_top5_g_gender_and_country.unstack().fillna(0)


fig, axs = plt.subplots(2, 1, figsize=(12, 10))
ax = (df_top5.query('Country == "India"')\
    .groupby(['Gender', 'year']).size()\
    .unstack()\
    .sort_index(ascending=False)\
    .plot(kind='bar',
          stacked=False,
          ax=axs[0],
          cmap='Purples',
          title='India'))
axs[0].legend(bbox_to_anchor=(1, 1))

ax = (df_top5.query('Country == "United States of America"')\
    .groupby(['Gender', 'year']).size()\
    .unstack()\
    .sort_index(ascending=False)\
    .plot(kind='bar',
          stacked=False,
          ax=axs[1],
          cmap='Purples',
          title='United States of America'))
axs[1].legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


df_top5['Education'] = df_top5['Education'].str.replace("â", "’")


df_top5_g_education = df_top5.groupby(['year', 'Country'])['Education'].value_counts(normalize=True)*100
df_top5_g_education.unstack().fillna(0)


name = [ 'Master’s degree', 'Bachelor’s degree','Doctoral degree', 'Some college/university study without earning a bachelor’s degree', 'I prefer not to answer', 'Professional degree', 'No formal education past high school','Professional doctorate']
value = df_top5[['Education']].value_counts(normalize=True)*100
value


fig = px.pie(
    df_top5, values=value, names=name,
    hole=0.5
)
fig.update_layout()
fig.show()


df_top5.query('Country == "India"')\
    .groupby(['Education', 'year']).size()\
    .unstack()\
    .sort_index(ascending=False)\
    .plot(kind='barh',
          stacked=False,
          cmap='Purples',
          title='India')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

df_top5.query('Country == "United States of America"')\
    .groupby(['Education', 'year']).size()\
    .unstack()\
    .sort_index(ascending=False)\
    .plot(kind='barh',
          stacked=False,
          cmap='Purples',
          title='United States of America')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
