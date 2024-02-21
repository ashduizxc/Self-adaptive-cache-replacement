import pandas as pd

a = 200000

df1 = pd.read_csv('mcf1.csv',nrows = a - 1,header = 0)

df1.to_csv('mcf1O'+'.csv',mode='a',header=1,index=0)
