from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)
model = tf.keras.models.load_model("fraud_model.h5")
scaler = joblib.load("scaler.save")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_features = [float(request.form[f]) for f in ["eps", "debt", "vol", "ceo"]]
    scaled_input = scaler.transform([input_features])
    prediction = model.predict(scaled_input)[0][0]
    result = "✅ Safe" if prediction < 0.5 else "⚠️ Fraud Risk"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
