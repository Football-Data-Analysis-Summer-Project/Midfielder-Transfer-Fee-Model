from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("midfielder.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Input feature names used during model training
feature_names = [
    "Age", "Club Level", "Minutes Played", "Goals", "Assists", "xG", "xA",
    "Progressive Passes", "Pass Completion", "SCA", "Interceptions", "Games Missed"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect inputs from form and convert to float
            inputs = [float(request.form[f]) for f in feature_names]
            # Scale using pre-trained scaler
            inputs_scaled = scaler.transform([inputs])
            # Predict using the model
            prediction = model.predict(inputs_scaled)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", feature_names=feature_names, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True,port=3050)
