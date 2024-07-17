from kafka import KafkaConsumer
import json

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'electricity_data',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Consumer started. Waiting for messages...")

for message in consumer:
    data = message.value
    print(f"Consumed: {data}")
    # Process data (e.g., send to ML model, save to blockchain, etc.)
