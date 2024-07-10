from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

app = Flask(__name__)

# Load the model
model = joblib.load('models/lgbm_modelv2.pkl')


# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)

        print("Received data:", data)

        # Extract the device ID and values
        device_id = data.get("id_devices", "unknown")
        values = data.get("data", {}).get("value", [])

        if not values:
            return jsonify({"error": "No data provided"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([values], columns=[str(i) for i in range(len(values))])

        # Normalize the data
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df)

        # Predict using the model
        model_predictions = model.predict(df_scaled)

        # Add threshold-based anomaly detection
        threshold_predictions = [1 if value > 4 else 0 for value in values]

        # Combine both predictions
        final_predictions = 1 if any(threshold_predictions) else int(model_predictions[0])

        # Create the response
        response = {
            "id_devices": device_id,
            "data": {
                "value": values,
                "hash": "null",
                "prediction": final_predictions
            },
            "time_stamp": int(time.time() * 1000),
            "status_code": 200
        }

        print("Response:", response)

        # Return the response as a JSON
        return jsonify(response)
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
