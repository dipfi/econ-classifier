import pandas as pd

data = {'first_column':  ['A', 'B', 'C'],
        'second_column': ['X','Y','Z']
        }

df = pd.DataFrame(data)

df["first_column"][df["second_column"]=="X"] = 0

print (df)

