import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
student_df = pd.read_csv("app/student_risk_predictions.csv")
mapping_df = pd.read_csv("app/advisor_student_mapping.csv")


# Prepare model and encoder
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features]
y = student_df["risk_level"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Streamlit UI
st.title("🎓 Student Risk Prediction Dashboard")

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
        risk_predictions = model.predict(X_filtered)
        filtered_df["Predicted Risk"] = le.inverse_transform(risk_predictions)

        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])
    else:
        st.warning("No students found for this user ID.")
