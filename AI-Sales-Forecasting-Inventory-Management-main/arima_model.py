import pandas as pd

train = pd.read_csv("data/train.csv")
features = pd.read_csv("data/features.csv")
stores = pd.read_csv("data/stores.csv")

# Merge datasets
df = train.merge(features, on=['Store', 'Date'], how='left')
df = df.merge(stores, on='Store', how='left')

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Aggregate sales
df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Rename
df.rename(columns={'Weekly_Sales': 'Sales'}, inplace=True)

sales = df['Sales']