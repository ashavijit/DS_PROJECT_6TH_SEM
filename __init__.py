import pandas as pd
df = pd.read_csv('winequality-red.csv', delimiter=';') # StackOverflow: https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python 

print(df.isnull().sum()) # Check for missing values

df = df.dropna()

print(df.dtypes)

# as dtype is object, we need to convert it to int
df['quality'] = df['quality'].astype(int)


df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()

df.to_csv('winequality_preprocessed.csv', index=False)
