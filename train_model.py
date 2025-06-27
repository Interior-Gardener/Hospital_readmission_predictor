import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("diabetic_data.csv")

# Drop irrelevant or high-missing columns
df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# Replace '?' with NaN and drop missing rows
df = df.replace('?', np.nan).dropna()

# Create binary target variable from 'readmitted'
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)
df.drop(['readmitted'], axis=1, inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 🔧 Fix column names for XGBoost compatibility
df.columns = (
    df.columns
    .str.replace(r"\[|\]|\<|\>", "", regex=True)
    .str.replace(r"\(|\)", "", regex=True)
    .str.replace(" ", "_")
)


# Prepare training data
X = df.drop('readmitted_binary', axis=1)
y = df['readmitted_binary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and feature columns
joblib.dump(model, "readmission_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

# SHAP explanation
X_test = X_test.astype(float)
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_plot.png")
print("\n✅ Model training complete. Files saved:")
print(" - readmission_model.pkl")
print(" - model_columns.pkl")
print(" - shap_summary_plot.png")
