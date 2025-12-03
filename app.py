from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained pipeline
# Ensure fertilizer_pipeline.pkl is in the same directory
try:
    model = joblib.load('fertilizer_pipeline.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return "Fertilizer Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Expecting JSON like:
        # {
        #   "Temparature": 26,
        #   "Humidity": 52,
        #   "Moisture": 38,
        #   "Soil Type": "Sandy",
        #   "Crop Type": "Maize",
        #   "Nitrogen": 37,
        #   "Potassium": 0,
        #   "Phosphorous": 0
        # }
        
        # Convert dict to DataFrame (required by the pipeline)
        input_df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Render assigns a PORT via environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
