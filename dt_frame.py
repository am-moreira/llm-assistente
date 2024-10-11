import pandas as pd

df = pd.read_csv('fidelizacao.csv', sep=';')
df = df.drop('OBSERVAÇÃO', axis=1)

print(df.head())

