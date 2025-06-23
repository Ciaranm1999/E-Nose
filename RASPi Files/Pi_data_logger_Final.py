import csv
import paho.mqtt.client as mqtt
from datetime import datetime
import time
import traceback
import socket  # For optional network check

# === CONFIG ===
MQTT_BROKER = "192.168.0.175"
MQTT_TOPIC = "mq3/data"
CSV_FILE = "/home/enose/Desktop/raw_data.csv"

# === WAIT FOR NETWORK ===
time.sleep(60)  # Wait 1 minute before starting to ensure network is ready

# Optionally check network availability before continuing
def wait_for_network(host="8.8.8.8", port=53, timeout=3):
    while True:
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return
        except Exception:
            print("Waiting for network...")
            time.sleep(5)

wait_for_network()  # Comment out if not needed

# === Ensure CSV file exists with header
try:
    with open(CSV_FILE, 'r'):
        pass
except FileNotFoundError:
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Message'])

def on_connect(client, userdata, flags, rc):
    try:
        if rc == 0:
            print("? Connected to MQTT Broker!")
            client.subscribe(MQTT_TOPIC)
            with open("/home/enose/Desktop/startup_log.txt", "a") as f:
                f.write(f"MQTT connected at {datetime.now()}\n")
        else:
            print(f"?? Failed to connect, return code {rc}")
            with open("/home/enose/Desktop/startup_log.txt", "a") as f:
                f.write(f"Failed MQTT connect: rc={rc} at {datetime.now()}\n")
    except Exception as e:
        with open("/home/enose/Desktop/startup_log.txt", "a") as f:
            f.write(f"Error in on_connect: {e}\n{traceback.format_exc()}\n")

def on_message(client, userdata, msg):
    try:
        log_msg = msg.payload.decode()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Received: {log_msg}")
        with open("/home/enose/Desktop/startup_log.txt", "a") as f:
            f.write(f"[{timestamp}] Received: {log_msg}\n")
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, log_msg])
    except Exception as e:
        with open("/home/enose/Desktop/startup_log.txt", "a") as f:
            f.write(f"Error in on_message: {e}\n{traceback.format_exc()}\n")

# === MQTT Setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, 1883)
except Exception as e:
    with open("/home/enose/Desktop/startup_log.txt", "a") as f:
        f.write(f"MQTT connection error: {e}\n{traceback.format_exc()}\n")
    raise

client.loop_forever()
