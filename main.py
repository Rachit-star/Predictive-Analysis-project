import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('adult_merged.csv')

print("First 5 rows of the merged dataset:")
print(df.head())

# --- IQR Outlier Removal START ---
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
# --- IQR Outlier Removal END ---

plt.scatter(df['age'], df['hours-per-week'], alpha=0.5)
plt.title('Age vs Hours-per-week (After Outlier Removal)')
plt.xlabel('Age')
plt.ylabel('Hours-per-week')
plt.savefig('scatter_plot.png')
print("Scatter plot saved as scatter_plot.png")
