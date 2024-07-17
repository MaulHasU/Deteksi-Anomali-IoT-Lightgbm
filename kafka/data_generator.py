from flask import Flask, jsonify, request
from kafka import KafkaProducer
import json
import time
import random

app = Flask(__name__)

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

@app.route('/generate', methods=['POST'])
def generate_data():
    device_id = request.json.get('device_id')
    frequency = request.json.get('frequency', 1)  # Frequency in seconds

    def generate_reading():
        return {
            "device_id": device_id,
            "timestamp": int(time.time() * 1000),
            "current": round(random.uniform(0.5, 10.0), 3)  # Simulated current values
        }

    while True:
        data = generate_reading()
        producer.send('electricity_data', data)
        print(f"Produced: {data}")
        time.sleep(frequency)

    return jsonify({"message": "Data generation started"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
