import pandas as pd
df=pd.read_csv('dst.csv')
zz=df.groupby('bid').mean()
print(zz)

zz.to_csv('result.csv')

