import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load('models/lgbm_modelv2.pkl')

# Sample data for testing
data = {
    "id_devices": "device_001",
    "data": {
        "value": [
            1.576, 2.843, 1.585, 2.581, 2.522, 1.483, 1.345, 2.081, 1.782, 1.956,
            2.439, 1.742, 2.123, 1.798, 2.352, 1.647, 2.768, 1.855, 2.442, 2.056,
            2.023, 1.983, 1.676, 2.481, 1.712, 5.944, 2.435, 1.873, 2.312, 1.889,
            1.845, 2.334, 1.752, 2.145, 1.955, 1.962, 1.857, 1.982, 1.748, 2.072,
            1.853, 1.936, 2.127, 1.842, 2.061, 1.993, 2.345, 1.879, 2.124, 1.867,
            2.176, 1.734, 2.231, 1.754, 2.341, 1.846, 2.129, 1.764, 2.049, 2.348,
            1.732, 2.193, 1.837, 2.129, 1.874, 2.213, 1.845, 2.035, 1.983, 2.112,
            1.756, 2.176, 1.845, 2.312, 1.763, 2.124, 6.873, 2.035, 1.962, 2.156,
            1.879, 2.193, 1.836, 2.123, 1.856, 2.147, 1.983, 2.048, 1.923, 2.137,
            1.853, 2.134, 1.836, 2.098, 1.923, 2.123, 1.873, 2.114, 1.895, 2.078,
            1.934, 2.123, 1.873, 2.034, 1.962, 2.111, 1.879, 2.195, 1.843, 2.034,
            1.893, 2.312, 1.862, 2.145, 1.895, 2.193, 1.836, 2.121, 1.852, 2.134,
            1.732, 2.322, 1.312, 2.422, 1.312, 2.927, 1.423, 2.132, 1.863, 2.423,
            1.127, 2.954, 1.654, 2.423, 1.673, 2.121, 1.431, 2.325, 1.893, 2.764,
            1.103, 2.091, 1.644, 2.958, 1.897, 2.432, 1.832, 2.287, 1.983, 2.849,
            7.820,
        ]
    }
}

# Extract the values
values = data["data"]["value"]

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

# Print the final prediction
print("Final Prediction:", final_predictions)
