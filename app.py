from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model = None
try:
    with open('fmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    if model is None:
        return jsonify({
            "status": "error", 
            "message": "Model not loaded. Check server logs."
        }), 500
    return jsonify({
        "status": "healthy", 
        "message": "Fertilizer Prediction API is running."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded on server"}), 500
    
    try:
        # Handle both JSON and Form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Input validation
        required_fields = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing fields: {missing}"}), 400

        # Prepare input data matching training columns
        input_data = {
            'Temparature': [int(data.get('Temparature'))],
            'Humidity': [int(data.get('Humidity'))],
            'Moisture': [int(data.get('Moisture'))],
            'Soil Type': [str(data.get('Soil Type')).strip()],
            'Crop Type': [str(data.get('Crop Type')).strip()],
            'Nitrogen': [int(data.get('Nitrogen'))],
            'Potassium': [int(data.get('Potassium'))],
            'Phosphorous': [int(data.get('Phosphorous'))]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except ValueError as e:
        return jsonify({"success": False, "error": "Invalid input type. Ensure numeric values are correct.", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
