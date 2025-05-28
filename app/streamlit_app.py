import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# Features
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

# --- Step 1: Generate synthetic risk labels ---
def generate_risk(row):
    score = (row["attendance_rate"] * 0.3 +
             row["gpa"] * 25 +
             row["assignment_completion"] * 0.2 +
             row["lms_activity"] * 0.3)
    if score >= 200:
        return "Low"
    elif score >= 140:
        return "Medium"
    else:
        return "High"

student_df["risk_level"] = student_df.apply(generate_risk, axis=1)

# --- Step 2: Train model using synthetic labels ---
X = student_df[features]
y = student_df["risk_level"]

# Encode labels
label_mapping = {"Low": 0, "Medium": 1, "High": 2}
label_inverse = {v: k for k, v in label_mapping.items()}
y_encoded = y.map(label_mapping)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Step 3: Streamlit UI ---
st.set_page_config(page_title="Student Risk Predictor", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard")

# ✅ Display model accuracy at the top
st.markdown("## ✅ Model Accuracy")
st.success(f"The model accuracy based on synthetic risk levels is **{accuracy * 100:.2f}%**")

# --- Role and ID Input ---
role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        X_filtered = filtered_df[features]
        predicted_risk = model.predict(X_filtered)
        filtered_df["Predicted Risk"] = [label_inverse[i] for i in predicted_risk]

        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])
    else:
        st.warning("No students found for this user ID.")
