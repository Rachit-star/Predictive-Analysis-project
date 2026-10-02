import pandas as pd

# Read the merged CSV file
df = pd.read_csv('adult_merged.csv')

# Print the first 5 rows
print("First 5 rows of the merged dataset:")
print(df.head())
