# libraries
import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Loading the data
data = sns.load_dataset('taxis')
display(data.head())


print(np.corrcoef(data.distance, data.total))



sns.scatterplot(x=data.distance, y=data.total)
plt.show()


print(st.pearsonr(data.distance, data.total))
print('----------------------------x------------------------', '\n')
print(st.spearmanr(data.distance, data.total))


sns.pairplot(data)
plt.show()


plt.figure(figsize=[12, 8])
sns.heatmap(data.corr())
plt.show()


display(data.corr())












