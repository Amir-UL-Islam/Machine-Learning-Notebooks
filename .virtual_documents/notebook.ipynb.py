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








































































