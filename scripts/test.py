import pandas as pd

# Load the preprocessed data
df = pd.read_csv('../data/processed/cleaned_data.csv')

# Check the column names
print(df.columns)
