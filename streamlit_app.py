import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load CSVs
student_df = pd.read_csv("app/student_risk_predictions.csv")
mapping_df = pd.read_csv("app/advisor_student_mapping.csv")

# Features used for prediction
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

# Generate sample training data (mocked with known labels)
# In production, replace with actual labeled training data
train_data = pd.DataFrame({
    "attendance_rate": [95, 60, 85, 50, 70],
    "gpa": [3.9, 2.1, 3.2, 1.9, 2.8],
    "assignment_completion": [100, 50, 90, 40, 60],
    "lms_activity": [95, 40, 75, 30, 50],
    "risk_level": ["low", "high", "medium", "high", "medium"]
})

# Prepare training data
X_train = train_data[features]
y_train = train_data["risk_level"]
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_encoded)

# Streamlit UI
st.set_page_config(page_title="Student Risk Prediction", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard")

role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    # Get students for advisor/chair
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        # Predict risk level
        X_filtered = filtered_df[features]
        risk_predictions = model.predict(X_filtered)
        filtered_df = filtered_df.copy()
        filtered_df["Predicted Risk"] = le.inverse_transform(risk_predictions)

        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])
    else:
        st.warning("No students found for this user ID.")
