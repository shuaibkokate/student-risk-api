from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Load data
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")

# Prepare features
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features]
y = student_df["risk_level"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

@app.get("/risk_prediction/")
def get_student_risk(user_id: str, role: str):
    if role not in ["advisor", "chair"]:
        raise HTTPException(status_code=400, detail="Role must be 'advisor' or 'chair'")

    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    elif role == "chair":
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if filtered_df.empty:
        return []

    # Predict risk
    X_filtered = filtered_df[features]
    risk_predictions = model.predict(X_filtered)
    filtered_df["predicted_risk"] = le.inverse_transform(risk_predictions)

    return filtered_df[["student_id", "attendance_rate", "gpa", "assignment_completion", "lms_activity", "predicted_risk"]].to_dict(orient="records")
