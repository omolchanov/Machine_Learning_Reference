import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd

# Define ontology-like relationships
disease_symptoms = {
    "Flu": ["Fever", "Cough", "Fatigue"],
    "Cold": ["Cough", "Sneezing", "Runny nose"],
    "Allergy": ["Sneezing", "Runny nose", "Itchy eyes"]
}

# Generate dataset (patients with symptoms)
patients = [
    {"name": "John", "symptoms": ["Fever", "Cough"], "disease": "Flu"},
    {"name": "Anna", "symptoms": ["Sneezing", "Runny nose"], "disease": "Cold"},
    {"name": "Bob", "symptoms": ["Sneezing", "Itchy eyes"], "disease": "Allergy"},
    {"name": "Mike", "symptoms": ["Cough", "Fatigue"], "disease": "Flu"},
    {"name": "Sara", "symptoms": ["Runny nose", "Cough"], "disease": "Cold"},
]

# Convert to ML features using ontology (bag of symptoms)
all_symptoms = sorted({s for syms in disease_symptoms.values() for s in syms})


def encode_symptoms(symptom_list):
    return [1 if s in symptom_list else 0 for s in all_symptoms]


X = [encode_symptoms(p["symptoms"]) for p in patients]
y = [p["disease"] for p in patients]

df = pd.DataFrame(X, columns=all_symptoms)
df["Disease"] = y
print(df)

# Train ML model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict a new case
new_patient = ["Fever", "Fatigue"]  # Symptoms
encoded = [encode_symptoms(new_patient)]
prediction = clf.predict(encoded)[0]
print(f"\nNew patient with {new_patient} is predicted to have: {prediction}")
