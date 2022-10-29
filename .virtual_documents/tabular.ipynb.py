import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sample_submission = pd.read_csv('datasets/tabular-playground-series-sep-2022/sample_submission.csv')
display(sample_submission.head())


test_df = pd.read_csv('datasets/tabular-playground-series-sep-2022/test.csv', parse_dates=['date'])
display(test_df.head())


train_df = pd.read_csv('datasets/tabular-playground-series-sep-2022/train.csv', parse_dates=['date'])
train_df = train_df.set_index('date').drop('row_id', axis=1)
display(train_df.head())
display(train_df.info())


display(train_df['product'].value_counts())


display(train_df['store'].value_counts())


display(train_df['country'].value_counts())


train_df.isnull().sum()


grouped = train_df.groupby(['country','store','product'])['num_sold'].mean()
grouped


grouped = train_df.groupby(['country','store'])['num_sold'].mean()
grouped


grouped = train_df.groupby(['country'])['num_sold'].mean()
grouped


plt.figure(figsize=(15,5))
train_df.groupby(['country','store','product'])['num_sold'].mean().unstack().plot(kind='bar',stacked=True)
plt.show()


import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Colors
color_pal = sns.color_palette()

plt.style.use('fivethirtyeight')


print(train_df.index.min())
print(train_df.index.max())



train = train_df.loc[train_df.index < '31-12-2019']
test = train_df.loc[train_df.index >= '31-12-2019']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('31-12-2019', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


def create_features(df):
    """
    Create time series features based on time series index.
    """
    train_df = df.copy()
#     train_df['hour'] = train_df.index.hour
    train_df['dayofweek'] = train_df.index.dayofweek
    train_df['quarter'] = train_df.index.quarter
    train_df['month'] = train_df.index.month
    train_df['year'] = train_df.index.year
    train_df['dayofyear'] = train_df.index.dayofyear
    train_df['dayofmonth'] = train_df.index.day
    train_df['weekofyear'] = train_df.index.isocalendar().week
    return train_df

train_df = create_features(train_df)
train_df.head()


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=train_df, x='month', y='num_sold', palette='Blues')
ax.set_title('M by Month')
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=train_df, x='year', y='num_sold', palette='Blues')
ax.set_title('M by Year')
plt.show()


X = train_df.drop('num_sold', axis=1)
y = train_df[['num_sold']]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()


# # train_df.head()
# datetime = pd.to_datetime(train_df['date'])
# print('Value Counts: ',datetime.value_counts().head())

# start = datetime.min()
# end = datetime.max()

# # Time Delta
# duration = end - start
# print('\n Total Sale Duration: ', duration)


train_df.head()
country = train_df['country'].unique()
country = list(country)

store = train_df['store'].unique()
store = list(store)

product = train_df['product'].unique()
product = list(product)



from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Encoding
pipeline = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(categories=[country, store, product]), ['country', 'store', 'product']),
    ('std_scaler', StandardScaler() , ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear'])
], remainder='passthrough')


X_train = pipeline.fit_transform(X_train)
X_train


X_test = pipeline.transform(X_test)
X_test.shape


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1500,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=18,
                       learning_rate=0.15,
                       random_state=42
                      )
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=1000, max_depth=9, random_state=42)



from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# grid search
model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1500,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=18,
                       random_state=42
                      )

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

param_grid = dict(learning_rate=learning_rate)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)

grid_result = grid_search.fit(X_train, y_train, eval_set=[(X_train_f, y_train_f)], verbose=100)


# grid search
model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
print("Best: get_ipython().run_line_magic("f", " using %s\" % (grid_result.best_score_, grid_result.best_params_))")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("get_ipython().run_line_magic("f", " (%f) with: %r\" % (mean, stdev, param))")
# plot
pyplot.errorbar(learning_rate, means, yerr=stds)
pyplot.title("XGBoost learning_rate vs Log Loss")
pyplot.xlabel('learning_rate')
pyplot.ylabel('Log Loss')
pyplot.savefig('learning_rate.png')



# im_f = pd.DataFrame(data=reg.feature_importances_,
#                     index=reg.feature_names_in_,
#                     columns=['importance'])

# im_f.sort_values('importance').plot(kind='barh', title='Feature Importance')
# # plt.show()


X_train_f = pipeline.transform(train_df)
y_train_f = train_df[['num_sold']].values

reg.fit(X_train_f, y_train_f,
        eval_set=[(X_train_f, y_train_f)],
        verbose=100)
# model.fit(X_train_f, y_train_f)



test_df_f = test_df.set_index('date').drop('row_id', axis=1)
test_df_f = create_features(test_df_f)
X_test_f = pipeline.transform(test_df_f)


prediction = reg.predict(X_test_f)


output = pd.DataFrame({'row_id': test_df.row_id, 'num_sold': prediction})
output.to_csv('submint1112.csv', index=False)
print("Successfullget_ipython().getoutput("")")
