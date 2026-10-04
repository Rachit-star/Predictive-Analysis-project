import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

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
plt.close()
print("Scatter plot saved as scatter_plot.png")

# CORRELATION START
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Heatmap saved as correlation_heatmap.png")

# TRAIN/TEST SPLIT START
X = df.drop('class', axis=1)
y = df['class']

# Using stratify=y to maintain the 75/25 class imbalance ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n--- Data Split Successfully ---")
print(f"Training data shape (X_train): {X_train.shape}")
print(f"Testing data shape (X_test): {X_test.shape}")
# TRAIN/TEST SPLIT END

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.8)]
df = df.drop(columns=to_drop)

print(f"Dropped highly correlated features: {to_drop}")
print(f"Final shape: {df.shape}")
# CORRELATION END

# MODEL TRAINING START
# Using a Pipeline as requested, and class_weight='balanced' for the 75/25 class imbalance!
pipeline = Pipeline([
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\n--- Logistic Regression (Pipeline) Model Evaluation ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# MODEL TRAINING END

# PREDICTING ON A COMPLETELY CUSTOM INPUT
print("\n--- Testing a Completely Custom Input ---")
# 1. Create a blank dataframe with the exact same 43 columns as our training data (filled with 0s)
custom_data = pd.DataFrame(0, index=[0], columns=X_train.columns)

# 2. Customize our completely fake person
# (Note: Numerical values like age are standardized, so 0.5 means slightly above average)
custom_data['age'] = 0.5 
custom_data['hours-per-week'] = 1.2
custom_data['education-num'] = 1.5
custom_data['sex'] = 1  
custom_data['capital-gain'] = 2.0  
custom_data['workclass_ Private'] = 1 
custom_data['marital-status_ Married-civ-spouse'] = 1 

# 3. Predict!
prediction = pipeline.predict(custom_data)
pred_label = ">50K" if prediction[0] == 1 else "<=50K"

print("Custom Person Profile:")
print("- Above average age, education, and hours-per-week")
print("- Male, Private Sector, Married")
print("- High Capital Gain")
print(f"\n---> Model Prediction for this custom person: {pred_label}")

# SAVE MODEL
joblib.dump(pipeline, 'model.pkl')
print("\nModel saved successfully as 'model.pkl'!")
