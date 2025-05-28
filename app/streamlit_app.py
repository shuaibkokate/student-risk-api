import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load student and mapping CSVs
student_df = pd.read_csv("app/student_risk_predictions.csv")
mapping_df = pd.read_csv("app/advisor_student_mapping.csv")

# --- Step 1: Training dataset (mock or real labeled)
train_data = pd.DataFrame({
    "attendance_rate": [95, 60, 85, 50, 70, 78, 90, 55, 40, 88],
    "gpa": [3.9, 2.1, 3.2, 1.9, 2.8, 3.0, 3.5, 2.3, 1.8, 3.4],
    "assignment_completion": [100, 50, 90, 40, 60, 70, 95, 45, 35, 88],
    "lms_activity": [95, 40, 75, 30, 50, 60, 90, 38, 28, 82],
    "risk_level": ["low", "high", "medium", "high", "medium", "medium", "low", "high", "high", "low"]
})

features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = train_data[features]
y = train_data["risk_level"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Streamlit UI ---
st.set_page_config(page_title="Student Risk Prediction", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard")

st.markdown(f"📈 **Model Accuracy:** `{accuracy * 100:.2f}%`")

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
        filtered_df = filtered_df.copy()
        filtered_df["Predicted Risk"] = le.inverse_transform(risk_predictions)

        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + ["Predicted Risk"]])
    else:
        st.warning("No students found for this user ID.")
