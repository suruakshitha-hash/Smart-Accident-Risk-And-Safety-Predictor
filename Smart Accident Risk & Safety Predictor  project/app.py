from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    vehicle = request.form['vehicle']
    gender = request.form['gender']
    speed = float(request.form['speed'])
    helmet = request.form.get('helmet', 'No')
    seatbelt = request.form.get('seatbelt', 'No')

    # Convert categorical values
    gender = 1 if gender == "Male" else 0
    helmet = 1 if helmet == "Yes" else 0
    seatbelt = 1 if seatbelt == "Yes" else 0

    # Vehicle logic
    if vehicle == "Car":
        helmet = 0
    elif vehicle == "Bike":
        seatbelt = 0

    # Prepare data for prediction
    data = [[age, gender, speed, helmet, seatbelt]]
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1] * 100  # survival probability

    # Determine result and risk
    if speed > 120 or (not seatbelt and not helmet):
        result = "Not Survived"
        risk = "High Risk 🔴"
        risk_class = "high"
        alert = "Emergency Alert! Immediate help required!"
    elif prediction == 1 and ((seatbelt and vehicle=="Car") or (helmet and vehicle=="Bike")) and speed < 80:
        result = "Survived"
        risk = "Low Risk 🟢"
        risk_class = "low"
        alert = "No emergency needed"
    else:
        result = "Not Survived"
        risk = "Medium Risk 🟡"
        risk_class = "medium"
        alert = "Caution advised"

    # Reasons for result
    reasons = []
    if speed > 80:
        reasons.append("High speed increases accident severity")
    if helmet == 0 and vehicle == "Bike":
        reasons.append("Helmet not used in bike")
    if seatbelt == 0 and vehicle == "Car":
        reasons.append("Seatbelt not used in car")
    if prediction == 1:
        reasons.append("Model predicts survival based on safe conditions")
    else:
        reasons.append("Model predicts high fatality risk")

    # Suggestions to improve safety
    suggestions = []
    if speed > 60:
        suggestions.append("Reduce speed below 60 km/h")
    if helmet == 0 and vehicle == "Bike":
        suggestions.append("Wear a helmet")
    if seatbelt == 0 and vehicle == "Car":
        suggestions.append("Use seatbelt")
    if not suggestions:
        suggestions.append("You are following all safety measures")

    # Render template with results
    return render_template(
        "index.html",
        prediction=result,
        risk=risk,
        risk_class=risk_class,
        alert=alert,
        prob=round(prob, 2),
        reasons=reasons,
        suggestions=suggestions
    )

if __name__ == "__main__":
    app.run(debug=True)