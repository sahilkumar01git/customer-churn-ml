import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("data/raw/telco_churn.csv")
df = df.drop(columns=["customerID"])

df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

object_columns = df.select_dtypes(include="object").columns
encoders = {}

for column in object_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le


X = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y
)


smote = SMOTE(random_state=50)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


models = {
    "Decision Tree": DecisionTreeClassifier(random_state=52),
    "Random Forest": RandomForestClassifier(random_state=52),
    "XGBoost": XGBClassifier(random_state=52, eval_metric="logloss", use_label_encoder=False)
}

for model_name, model in models.items():
    scores = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring="f1")
    print(f"{model_name} mean F1-score: {scores.mean():.2f}")


rfc = RandomForestClassifier(random_state=52)
rfc.fit(x_train_smote, y_train_smote)

y_test_pred = rfc.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))


import os
import pickle

os.makedirs("models", exist_ok=True)

model_data = {
    "model": rfc,
    "feature_names": X.columns.tolist()
}

with open("models/customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Model and encoders saved successfully.")