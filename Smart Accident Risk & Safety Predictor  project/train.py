import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("accident.csv")
df = df.dropna()

df['Gender']       = df['Gender'].map({'Male': 1, 'Female': 0})
df['Helmet_Used']  = df['Helmet_Used'].map({'Yes': 1, 'No': 0})
df['Seatbelt_Used']= df['Seatbelt_Used'].map({'Yes': 1, 'No': 0})

X = df[['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']]
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit only on training data
X_test_scaled  = scaler.transform(X_test)        # only transform on test data

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("model.pkl and scaler.pkl saved!")
