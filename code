import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("banking_transactions (1).csv")

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

df = df.dropna()

encoder = LabelEncoder()

categorical_columns = [
    'payment_channel',
    'authentication_type'
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print(df.head())

plt.figure(figsize=(6,4))

df['fraud_flag'].value_counts().plot(kind='bar')

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraud Flag")
plt.ylabel("Count")

plt.show()

X = df.drop(['fraud_flag', 'transaction_id'], axis=1)

y = df['fraud_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,5))

importance.plot(kind='bar')

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.show()

sample_transaction = X.iloc[0:1]

prediction = model.predict(sample_transaction)

print("\nSample Transaction Prediction:")

if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Normal Transaction")