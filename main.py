import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# OUTLIER REMOVAL START
iqr_cols = ['age', 'fnlwgt', 'hours-per-week']
Q1 = df[iqr_cols].quantile(0.25)
Q3 = df[iqr_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[iqr_cols] < (Q1 - 1.5 * IQR)) | (df[iqr_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
# OUTLIER REMOVAL END

# STANDARDIZATION START
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# STANDARDIZATION END

print(f"Shape: {df.shape}")
print(df.head())

plt.scatter(df['age'], df['hours-per-week'], alpha=0.5)
plt.title('Age vs Hours-per-week (After Outlier Removal)')
plt.xlabel('Age')
plt.ylabel('Hours-per-week')
plt.savefig('scatter_plot.png')
print("Scatter plot saved as scatter_plot.png")
