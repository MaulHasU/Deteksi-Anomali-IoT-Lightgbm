from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

app = Flask(__name__)

# Load the model
model = joblib.load('models/lgbm_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        device_id = data.get("id_devices", "unknown")
        values = data.get("data", {}).get("value", [])

        if not values:
            return jsonify({"error": "No data provided"}), 400

        df = pd.DataFrame([values], columns=[str(i) for i in range(len(values))])
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df)
        model_predictions = model.predict(df_scaled)
        threshold_predictions = [1 if value > 4 else 0 for value in values]
        final_predictions = 1 if any(threshold_predictions) else int(model_predictions[0])

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
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
