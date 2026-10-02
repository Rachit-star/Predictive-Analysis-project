import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('adult_merged.csv')

df = df.replace(['?', ' ?'], np.nan)
df['class'] = df['class'].str.replace('.', '', regex=False)

# IMPUTATION START
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# IMPUTATION END

# ENCODING START
df = df.drop('education', axis=1)

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['class'] = le.fit_transform(df['class'])
df['native-country'] = le.fit_transform(df['native-country'])

cols_to_ohe = ['workclass', 'marital-status', 'occupation', 'relationship', 'race']
df = pd.get_dummies(df, columns=cols_to_ohe, drop_first=True)
# ENCODING END

print(f"Shape: {df.shape}")
print(df.head())
