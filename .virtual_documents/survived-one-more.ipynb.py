import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_df = pd.read_csv('datasets/titanic/train.csv')
display(train_df.head())



women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("{} of women who survived:".format(rate_women))


men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("{} of men who survived:".format(rate_men))


train_df.hist(bins=50, figsize=(20, 15))
plt.show()


# print(train_df['Fare'].describe())
# print(np.quantile(train_df['Fare'], [ 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]))
# print(train_df['Fare'].value_counts().head(10))
train_df['Fare_Group'] = pd.cut(train_df['Fare'], bins=[0., 8., 20., 40., 50., 60., 75., np.inf], labels=[1, 2, 3, 4, 5, 6, 7])
train_df.head()


train_df['Age'].describe()
train_df['Age'].unique()


train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[0., 5., 10., 15., 25., 35., 45., 55., 65., 75., 85., np.inf], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


test_df = pd.read_csv('datasets/titanic/test.csv')
test_df['Age_Group'] = pd.cut(test_df['Age'], bins=[0., 5., 10., 15., 25., 35., 45., 55., 65., 75., 85., np.inf], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
test_df['Fare_Group'] = pd.cut(test_df['Fare'], bins=[0., 10., 20., 40., 50., 60., 75., np.inf], labels=[1, 2, 3, 4, 5, 6, 7])



from sklearn.ensemble import RandomForestClassifier

y = train_df["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age_Group", "Fare_Group"]
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

model = RandomForestClassifier(n_estimators=70, max_depth=7, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submint08.csv', index=False)
print("Successfullget_ipython().getoutput("")")
