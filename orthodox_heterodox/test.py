import pandas as pd

testdf = pd.DataFrame({"a": [1,2,3,4,5,6,7,8], "b": [1,1,1,1,2,2,2,2]})

testdf.loc[testdf["b"]==1]["a"]