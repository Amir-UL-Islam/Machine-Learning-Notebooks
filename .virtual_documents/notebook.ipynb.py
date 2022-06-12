import numpy as np
import pandas as pd

df1 = pd.DataFrame({'pointer':['A', 'B', 'C', 'B', 'A', 'D'], 
                    'value_df1':[0,1,2,3,4,5]})

df2 = pd.DataFrame({'pointer':['B', 'C', 'B', 'D', 'E'], 
                    'value_df2':[6, 7, 8, 9, 12]})

display(df1)
display(df2)

print("Left Merged DataFrame\n")

display(pd.merge(df1, df2, how = 'left')) # Performing a left merge


import numpy as np
import pandas as pd

df1 = pd.DataFrame({'pointer':['A', 'B', 'C', 'B', 'A', 'D'], 
                    'value_df1':[0,1,2,3,4,5]})

df2 = pd.DataFrame({'pointer':['B', 'Z', 'C', 'B','D','E'], 
                    'value_df2':[6,7,8,9,10,11]})

display(df1)
display(df2)


print("Right Merged DataFrame\n")
display(pd.merge(df1, df2, how = 'right')) # Performing a right merge


import numpy as np
import pandas as pd

df1 = pd.DataFrame({'pointer':['A', 'B', 'C', 'B', 'A', 'D'], 
                    'value_df1':[0,1,2,3,4,5]})

df2 = pd.DataFrame({'pointer':['B', 'Z', 'C', 'B','D','E'], 
                    'value_df2':[6,7,8,9,10,11]})

display(df1)
display(df2)

print("Outer Merged DataFrame\n")
display(pd.merge(df1, df2, how = 'outer')) # Performing an outer merge


import numpy as np
import pandas as pd

df1 = pd.DataFrame({'column1':['Pak', 'USA', 'Pak', 'UK', 'Ind','None'], #Column 1
                    'column2':['A', 'B', 'C', 'B', 'A', 'D'],            #Column 2
                    'value_df1':[0,1,2,3,4,5]})

df2 = pd.DataFrame({'column1':['USA', 'UK', 'None', 'USA', 'Pak','Ind'], #Column 1
                    'column2':['B', 'Z', 'C', 'B','D','E'],              #Column 2
                    'value_df2':[6,7,8,9,10,11]})

display(df1)
display(df2)

print("Outer Merged DataFrame on Multiple Columns\n")
display(pd.merge(df1, df2, on = ['column1', 'column2'], how = 'outer'))


import numpy as np
import pandas as pd

df1 = pd.DataFrame({'pointer':['A', 'B', 'C', 'B', 'A', 'D'], 
                    'value_df1':[0,1,2,3,4,5]})

df2 = pd.DataFrame(np.arange(10,13,1), index = ['A', 'B','C'], columns = ['values'])

display(df1)
display(df2)

print("Merged on index\n")
print(pd.merge(df1, df2, left_on='pointer', right_index=True))


import pandas as pd

df = pd.DataFrame({'City':['Lahore', 'Mumbai', 'Karachi', 'London'],
                   'AQI':[147, 166, 152, 81]})

print("The Original DataFrame")
display(df)

dict_map = {'Lahore':'Pakistan', 'Karachi':'Pakistan', 'Mumbai':'India', 'London':'UK'}

df['Country'] = df['City'].map(dict_map)

print("The Mapped DataFrame")
display(df)


import pandas as pd

df = pd.DataFrame({'Col1':['A', 'B', 'A', 'C', 'B', 'C'],
                    'Col2': [1, 2, 1, 3, 4, 3]})

print("The Original DataFrame")
display(df)

print("The DataFrame without duplicates")
display(df.drop_duplicates())

print("The DataFrame without Column1 duplicates")
display(df.drop_duplicates(['Col1']))


import numpy as np
import pandas as pd

df = pd.DataFrame(abs(np.random.randn(9)).reshape(3,3), 
                          index = ['row1', 'row2', 'row3'],
                          columns = ['col1', 'col2', 'col3'])
print("The original DataFrame\n")
display(df)
print("All row and column indexes are changed")
display(df.rename(index = str.upper, columns = str.title))


print("Specific row and column indexes are changed")
display(df.rename(index = {'row3':'row_index3'}, columns = {'col3':'col_index3'}))




import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(900,3))

display(df)

quantiles_df = (df.quantile([0.25,0.75]))

print("The 1st & 3rd quartiles of all columns:")
display(quantiles_df)




Q1 = quantiles_df[0][0.25]
Q3 = quantiles_df[0][0.75]


iqr = Q3 - Q1


lower_bound = (Q1 - (iqr * 1.5))
upper_bound = (Q3 + (iqr * 1.5))

print("The Lower bound for the first column:")
print(lower_bound, '\n')

print("The Upper bound for the first column:")
print(upper_bound, '\n')

col1 = df[0]

print("The outliers in the first column below the lower bound:")
print(col1[(col1 < lower_bound)])

print('\n', "The outliers in the first column above the upper bound:")
print(col1[(col1 > upper_bound)])


col1[(col1 < lower_bound)] = lower_bound
col1[(col1 > upper_bound)] = upper_bound

print("The outliers in the first column below the lower bound:")
print(col1[(col1 < lower_bound)])

print('\n', "The outliers in the first column above the upper bound:")
print(col1[(col1 > upper_bound)])


import numpy as np
import pandas as pd

df = pd.DataFrame({'p1':['A','A','B','B','C','C'],'p2':['G1','G2','G1','G2','G1','G2'],
    'val_1':np.random.randn(6),'val_2':np.random.randn(6)})

display(df) 

print("DataFrame after using groupby")
print(df.groupby('p1'))


import numpy as np
import pandas as pd

df = pd.DataFrame({'p1':['A','A','B','B','C','C'],'p2':['G1','G2','G1','G2','G1','G2'],
    'val_1':np.arange(1,7,1),'val_2':np.arange(7,13,1)})

print("The original DataFrame")
display(df)

print("DataFrame after using groupby on p1 and summing the values")
display(df.groupby('p1').sum())

print("DataFrame after using groupby on p2 and summing the values")
display(df.groupby('p2').sum())


import numpy as np
import pandas as pd

df = pd.DataFrame({'p1':['A','A','B','B','C','C'],'p2':['G1','G2','G1','G2','G1','G2'],
    'val_1':np.arange(1,7,1),'val_2':np.arange(7,13,1)})

print("The original DataFrame")
display(df)

print("DataFrame after using groupby on p1 & p2 and Summing their values")
display(df.groupby(['p1','p2']).sum())


import numpy as np
import pandas as pd
import random
# Declaring a DataFrame with values
df = pd.DataFrame({'Animal_type':[random.choice(['Chicken','Duck', 'Goat', 'Turkey']) for i in range(1,16)],
                   'legs':[random.choice(range(1,4)) for i in range(1,16)],
                   'weight':[random.choice(range(10,20)) for i in range(1,16)],
                   'height':[random.choice(range(4,15)) for i in range(1,16)],
                   'protein':abs(np.random.randn(15)),
                    })
print("The Original DataFrame:")
display(df)

Aw = df.groupby('Animal_type') # Grouping with Animal_type column

# Computing mean of individual groups
print("Average properties an animal can have:")
display(Aw.agg('mean'))
















