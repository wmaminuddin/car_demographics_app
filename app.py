from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
age_model = joblib.load('models/age_model.pkl')
gender_model = joblib.load('models/gender_model.pkl')
race_model = joblib.load('models/race_model.pkl')
marital_model = joblib.load('models/marital_model.pkl')

# Load encoders and scaler
gender_encoder = joblib.load('models/gender_encoder.pkl')
race_encoder = joblib.load('models/race_encoder.pkl')
marital_encoder = joblib.load('models/marital_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    height = float(request.form.get('height'))
    length = float(request.form.get('length'))
    width = float(request.form.get('width'))
    weight = float(request.form.get('weight'))
    seats = float(request.form.get('seats'))
    cargo = float(request.form.get('cargo'))
    price = float(request.form.get('price'))
    
    # Calculate volume
    volume = height * length * width
    
    # Prepare input features
    input_features = np.array([[height, length, width, volume, weight, seats, cargo, price]])
    input_scaled = scaler.transform(input_features)
    
    # Predict age
    age_pred = age_model.predict(input_scaled)[0]
    
    # Predict gender with confidence levels
    gender_proba = gender_model.predict_proba(input_scaled)[0]
    gender_confidence = {gender_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                        for i, prob in enumerate(gender_proba)}
    gender_pred = gender_encoder.inverse_transform([gender_proba.argmax()])[0]
    
    # Predict race with confidence levels
    race_proba = race_model.predict_proba(input_scaled)[0]
    race_confidence = {race_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                      for i, prob in enumerate(race_proba)}
    race_pred = race_encoder.inverse_transform([race_proba.argmax()])[0]
    
    # Predict marital status with confidence levels
    marital_proba = marital_model.predict_proba(input_scaled)[0]
    marital_confidence = {marital_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                         for i, prob in enumerate(marital_proba)}
    marital_pred = marital_encoder.inverse_transform([marital_proba.argmax()])[0]
    
    # Prepare results for display
    results = {
        'car_specs': {
            'height': height,
            'length': length,
            'width': width,
            'volume': volume,
            'weight': weight,
            'seats': seats,
            'cargo': cargo,
            'price': price
        },
        'age': {
            'prediction': round(age_pred, 1)
        },
        'gender': {
            'prediction': gender_pred,
            'confidence': gender_confidence
        },
        'race': {
            'prediction': race_pred,
            'confidence': race_confidence
        },
        'marital_status': {
            'prediction': marital_pred,
            'confidence': marital_confidence
        }
    }
    
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
