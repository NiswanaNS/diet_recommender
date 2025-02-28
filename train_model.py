import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("cleaned_diet_data.csv")

# Ensure correct columns
expected_columns = ["Age", "Gender", "Height_cm", "Weight_kg", "BMI", "Diet_Recommendation"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Convert numeric columns
numeric_columns = ["Age", "Height_cm", "Weight_kg", "BMI"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Fill missing values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Encode Gender
label_encoders = {}
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
label_encoders["Gender"] = le_gender

# Encode target column
le_diet = LabelEncoder()
df["Diet_Recommendation"] = le_diet.fit_transform(df["Diet_Recommendation"])
label_encoders["Diet_Recommendation"] = le_diet

# Select features and target
X = df[["Age", "Gender", "Height_cm", "Weight_kg", "BMI"]]
y = df["Diet_Recommendation"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save trained model
with open("diet_model.sav", "wb") as model_file:
    pickle.dump(model, model_file)

# Save label encoders
with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)

print("✅ Model and label encoders saved successfully!")
