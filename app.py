from flask import Flask, request, jsonify 
import pickle
import numpy as np
from datetime import datetime

# Create a Flask web application
app = Flask(__name__)

# Load the trained model from the .pkl file
with open('XGBRegressor.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the StandardScaler from the .pkl file
with open('standar_scalation.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Health check or welcome route (GET /)
@app.route('/', methods=['GET'])
def home():
    # Return a welcome message to confirm the API is running
    return jsonify({
        "message": "Welcome to the Sales Prediction API. Use POST /predict with the required fields."
    })

# Define the API endpoint: /predict that accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Try to extract JSON data from the request
        data = request.get_json()

        # If JSON is missing or invalid
        if data is None:
            return jsonify({
                "error": "Invalid or missing JSON payload"
            }), 400  # Return HTTP 400 Bad Request

        # Define required input fields
        required_fields = [
            "store_ID", "day_of_week", "date", "nb_customers_on_day",
            "open", "promotion", "state_holiday", "school_holiday"
        ]

        # Check if any required field is missing
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing_fields
            }), 400

        # Validate types and values for all fields
        if not isinstance(data["store_ID"], int) or data["store_ID"] < 0:
            return jsonify({"error": "store_ID must be a positive integer"}), 400

        if not isinstance(data["day_of_week"], int) or not 1 <= data["day_of_week"] <= 7:
            return jsonify({"error": "day_of_week must be an integer between 1 and 7"}), 400

        if not isinstance(data["date"], str):
            return jsonify({"error": "date must be a string in format DD/MM/YYYY"}), 400

        if not isinstance(data["nb_customers_on_day"], int) or data["nb_customers_on_day"] < 0:
            return jsonify({"error": "nb_customers_on_day must be a positive integer"}), 400

        if not isinstance(data["open"], int) or data["open"] not in [0, 1]:
            return jsonify({"error": "open must be 0 or 1"}), 400

        if not isinstance(data["promotion"], int) or data["promotion"] not in [0, 1]:
            return jsonify({"error": "promotion must be 0 or 1"}), 400

        if not isinstance(data["school_holiday"], int) or data["school_holiday"] not in [0, 1]:
            return jsonify({"error": "school_holiday must be 0 or 1"}), 400

        if not isinstance(data["state_holiday"], str) or data["state_holiday"] not in ["0", "a", "b", "c"]:
            return jsonify({
                "error": "state_holiday must be one of: '0', 'a', 'b', 'c'"
            }), 400

        # Business rule: if store is closed or has no customers, sales must be 0
        if data["open"] == 0 or data["nb_customers_on_day"] == 0:
            return jsonify({
                "prediction": 0.0,
                "message": "Store is closed or has no customers. Sales prediction is set to 0."
            })

        # Feature engineering: convert date string to ordinal number
        try:
            date_ordinal = datetime.strptime(data["date"], "%d/%m/%Y").toordinal()
        except ValueError:
            return jsonify({"error": "date must be in format DD/MM/YYYY"}), 400

        # One-hot encode day_of_week (we dropped DOW_1 during training)
        dow_encoded = [0] * 6  # Always 6 features for DOW_2 to DOW_7
        if 2 <= data["day_of_week"] <= 7:
            dow_encoded[data["day_of_week"] - 2] = 1
        # If day_of_week == 1 (Monday), leave all zeros (DOW_1 was dropped during training)

        # One-hot encode state_holiday ('0', 'a', 'b', 'c')
        state_map = {
            "0": [0, 0, 0],  # No holiday
            "a": [1, 0, 0],  # Public holiday
            "b": [0, 1, 0],  # Easter holiday
            "c": [0, 0, 1]   # Christmas holiday
        }
        state_encoded = state_map.get(data["state_holiday"], [0, 0, 0])  # Default if unknown

        # Build the input vector for the model
        # Must match the order and number of features used in training
        input_vector = [
            date_ordinal,
            data["nb_customers_on_day"],
            data["promotion"],
            data["school_holiday"],
            data["open"] 
        ] + state_encoded + dow_encoded  # Concatenate the encoded features

        # Apply the same scaler used during training
        input_scaled = scaler.transform([input_vector]) 

        # Make the prediction using the model
        prediction = model.predict(input_scaled)[0]  # Get single prediction

        # Check for low number of customers and include a warning if needed
        if data["nb_customers_on_day"] < 200:
            return jsonify({
                "prediction": round(float(prediction), 2),
                "warning": "Very few customers on this day. Prediction may be unreliable."
            })

        # Return the prediction in JSON format
        return jsonify({
            "prediction": round(float(prediction), 2)  # Round result
        })

    except Exception as e:
        # If any unexpected error happens, return error info
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500  # HTTP 500 Server Error

# Run the Flask app publicly on port 5000 (0.0.0.0 means visible from internet)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
