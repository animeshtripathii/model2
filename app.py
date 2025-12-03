from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import pickle
import logging

app = Flask(__name__)

# Enable CORS for all routes
# This allows any domain to access your API. 
# For production, you might want: CORS(app, origins=["https://your-frontend.com"])
CORS(app)

# Configure logging to see errors in Render dashboard
logging.basicConfig(level=logging.INFO)

# Load the trained model with robust error handling
model = None
try:
    with open('fmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file 'fmodel.pkl' not found.")
except Exception as e:
    # This catches the specific error causing your crash (e.g., missing scikit-learn)
    logging.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def home():
    if model is None:
        return jsonify({
            "status": "error", 
            "message": "Model not loaded. Check server logs for details."
        }), 500
    return jsonify({
        "status": "healthy", 
        "message": "Fertilizer Prediction API is running with CORS enabled."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    
    try:
        # Handle both JSON and Form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Validate inputs
        required_fields = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400

        # Prepare input data
        input_data = {
            'Temparature': [int(data.get('Temparature'))],
            'Humidity': [int(data.get('Humidity'))],
            'Moisture': [int(data.get('Moisture'))],
            'Soil Type': [data.get('Soil Type')],
            'Crop Type': [data.get('Crop Type')],
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
        return jsonify({"success": False, "error": "Invalid input type. Ensure numbers are correct.", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
