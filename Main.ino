#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_BME680.h>

const char* ssid = "TP-Link_20EF";
const char* password = "64052130";

const char* mqtt_server = "192.168.0.175";
const int mqtt_port = 1883;
const char* mqtt_topic = "mq3/data";

WiFiClient espClient;
PubSubClient mqttClient(espClient);
Adafruit_BME680 bme;

const int MQ3_A_PINS[3] = {32, 35, 34};
const int MQ3_D_PINS[3] = {25, 26, 27};

const int LED_PIN = 2;
bool bmeAvailable = false;

const unsigned long PUBLISH_INTERVAL = 10000; // 1 minute
const unsigned long RESET_INTERVAL = 15UL * 60UL * 1000UL; // 15 minutes
unsigned long lastPublishTime = 0;
unsigned long startTime = 0;

void flashLED(int times, int duration) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(duration);
    digitalWrite(LED_PIN, LOW);
    delay(duration > 200 ? duration : 100);
  }
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  digitalWrite(LED_PIN, HIGH);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  digitalWrite(LED_PIN, LOW);
  Serial.println("\n✅ WiFi connected!");
  Serial.print("   IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (mqttClient.connect("ESP32Client")) {
      Serial.println("reconnected!");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  setup_wifi();
  mqttClient.setServer(mqtt_server, mqtt_port);
  Serial.println("🔌 Attempting to connect to MQTT broker...");
  if (mqttClient.connect("ESP32Client")) {
    Serial.println("✅ MQTT connected!");
    flashLED(1, 1000);
  } else {
    Serial.print("❌ MQTT connection failed, rc=");
    Serial.println(mqttClient.state());
    flashLED(5, 150);
  }

  Wire.begin(21, 22);

  Serial.println("🔍 Initializing BME680...");
  if (bme.begin(0x76) || bme.begin(0x77)) {
    bmeAvailable = true;
    Serial.println("✅ BME680 detected!");
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setGasHeater(320, 150);
  } else {
    Serial.println("❌ BME680 not found. Continuing without it...");
  }

  analogReadResolution(12);
  for (int i = 0; i < 3; i++) {
    analogSetPinAttenuation(MQ3_A_PINS[i], ADC_11db);
    pinMode(MQ3_D_PINS[i], INPUT);
  }

  lastPublishTime = millis();
  startTime = millis();
}

void loop() {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }
  mqttClient.loop();

  unsigned long currentTime = millis();

  // Time to publish?
  if (currentTime - lastPublishTime >= PUBLISH_INTERVAL) {
    float temp = 0.0, humidity = 0.0, pressure = 0.0, gas = 0.0;
    bool bmeReadingSuccess = false;

    if (bmeAvailable) {
      if (bme.performReading()) {
        temp = bme.temperature;
        humidity = bme.humidity;
        pressure = bme.pressure / 100.0;
        gas = bme.gas_resistance / 1000.0;
        bmeReadingSuccess = true;
      }
    }

      char jsonPayload[512];

  // Start the JSON string manually
  strcpy(jsonPayload, "{\"bme680\":{");
  snprintf(jsonPayload + strlen(jsonPayload), sizeof(jsonPayload) - strlen(jsonPayload),
    "\"available\":%s,\"reading_ok\":%s,\"temperature\":%.2f,\"humidity\":%.2f,\"pressure\":%.2f,\"gas_kOhm\":%.2f},\"mq3\":[",
    bmeAvailable ? "true" : "false",
    bmeReadingSuccess ? "true" : "false",
    temp, humidity, pressure, gas);

  // Append only valid MQ3 sensor readings
  bool firstEntry = true;
  for (int i = 0; i < 3; i++) {
    int analogVal = analogRead(MQ3_A_PINS[i]);
    int digitalVal = digitalRead(MQ3_D_PINS[i]);
    
    if (analogVal > 0 && analogVal < 4095) {
      if (!firstEntry) {
        strncat(jsonPayload, ",", sizeof(jsonPayload) - strlen(jsonPayload) - 1);
      }
      char entry[100];
      snprintf(entry, sizeof(entry),
              "{\"id\":%d,\"analog\":%d,\"digital\":%d}",
              i + 1, analogVal, digitalVal);
      strncat(jsonPayload, entry, sizeof(jsonPayload) - strlen(jsonPayload) - 1);
      firstEntry = false;
    }
  }

  // Close JSON object
  strncat(jsonPayload, "]}", sizeof(jsonPayload) - strlen(jsonPayload) - 1);


    Serial.print("Publishing to topic: ");
    Serial.println(mqtt_topic);
    Serial.println(jsonPayload);
    mqttClient.publish(mqtt_topic, jsonPayload);
    Serial.println("----------------------------\n");

    lastPublishTime = currentTime;
  }

  // Check if it's time to restart
  if (currentTime - startTime >= RESET_INTERVAL) {
    Serial.println("🔁 Performing scheduled soft reset.");
    delay(1000);
    esp_restart();
  }
}
