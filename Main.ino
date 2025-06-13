#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_BME680.h>

// WiFi and MQTT configuration
const char* ssid = "TP-Link_20EF";
const char* password = "64052130";
const char* mqtt_server = "192.168.0.175";
const int mqtt_port = 1883;
const char* mqtt_topic = "mq3/data";

WiFiClient espClient;
PubSubClient mqttClient(espClient);
Adafruit_BME680 bme;

// MQ-3 pin configuration
const int MQ3_BOTTOM_PIN = 35;
const int MQ3_BOTTOM_D_PIN = 26;
const int MQ3_TOP_PIN = 34;
const int MQ3_TOP_D_PIN = 25;
const int MQ3_A_PINS[3] = {32, MQ3_BOTTOM_PIN, MQ3_TOP_PIN};
const int MQ3_D_PINS[3] = {25, MQ3_BOTTOM_D_PIN, MQ3_TOP_D_PIN};

// Calibration & gas sensor constants
const float RL_VALUE = 10.0;
const float Vc = 5.0;
const float CLEAN_AIR_FACTOR = 60.0;

// Calibration control
const bool FORCE_CALIBRATION = false;
const int CALIBRATION_SAMPLES = 20;
const int CALIBRATION_INTERVAL = 250;

// R0 values (update these after calibration)
float R0_MQ3_BOTTOM = 1.87;  // ← Update after initial calibration
float R0_MQ3_TOP = 1.87;     // ← Update after initial calibration

const int LED_PIN = 2;
bool bmeAvailable = false;
const unsigned long PUBLISH_INTERVAL = 60000;
const unsigned long RESET_INTERVAL = 15UL * 60UL * 1000UL;
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

float readSensorResistance(int sensorPin) {
  int analogValue = analogRead(sensorPin);
  if (analogValue < 1) analogValue = 1;
  float sensorVoltage = analogValue * (Vc / 4095.0);
  if (sensorVoltage < 0.1) sensorVoltage = 0.1;
  return ((Vc - sensorVoltage) / sensorVoltage) * RL_VALUE;
}

float calibrateSensor(int sensorPin) {
  float rs_sum = 0.0;
  int valid_samples = 0;
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    float rs = readSensorResistance(sensorPin);
    if (rs > 0.1 && rs < 1000) {
      rs_sum += rs;
      valid_samples++;
    }
    delay(CALIBRATION_INTERVAL);
  }
  if (valid_samples < CALIBRATION_SAMPLES / 2) {
    Serial.println("⚠️ Too few valid samples, returning default R0 = 10");
    return 10.0;
  }
  float rs_avg = rs_sum / valid_samples;
  float r0 = rs_avg / CLEAN_AIR_FACTOR;
  if (r0 < 0.1 || r0 > 100) {
    Serial.println("⚠️ R0 out of expected range, returning default R0 = 10");
    return 10.0;
  }
  return r0;
}

float analogToPPM(int analogValue, float r0) {
  if (analogValue < 1) analogValue = 1;
  float sensorVoltage = analogValue * (Vc / 4095.0);
  if (sensorVoltage < 0.1) sensorVoltage = 0.1;
  float rs = ((Vc - sensorVoltage) / sensorVoltage) * RL_VALUE;
  float ratio = rs / r0;
  return 607.9 * pow(ratio, -2.868);
}

void setup_wifi() {
  delay(10);
  Serial.print("Connecting to "); Serial.println(ssid);
  digitalWrite(LED_PIN, HIGH);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  digitalWrite(LED_PIN, LOW);
  Serial.println("\n✅ WiFi connected! IP: " + WiFi.localIP().toString());
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT...");
    if (mqttClient.connect("ESP32Client")) {
      Serial.println("connected!");
    } else {
      Serial.print("failed (rc=");
      Serial.print(mqttClient.state());
      Serial.println("), retrying...");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(LED_PIN, OUTPUT);
  pinMode(MQ3_BOTTOM_D_PIN, INPUT);
  pinMode(MQ3_TOP_D_PIN, INPUT);

  setup_wifi();
  mqttClient.setServer(mqtt_server, mqtt_port);
  if (mqttClient.connect("ESP32Client")) {
    Serial.println("✅ MQTT connected!");
    flashLED(1, 1000);
  } else {
    Serial.print("❌ MQTT failed, rc=");
    Serial.println(mqttClient.state());
    flashLED(5, 150);
  }

  Wire.begin(21, 22);

  Serial.println("🔍 Initializing BME680...");
  if (bme.begin(0x76) || bme.begin(0x77)) {
    bmeAvailable = true;
    Serial.println("✅ BME680 detected.");
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setGasHeater(320, 150);
  } else {
    Serial.println("❌ BME680 not found.");
    bmeAvailable = false;
  }

  analogReadResolution(12);
  for (int i = 0; i < 3; i++) {
    analogSetPinAttenuation(MQ3_A_PINS[i], ADC_11db);
    pinMode(MQ3_D_PINS[i], INPUT);
  }

  // Calibrate MQ3 sensors only if forced
  if (FORCE_CALIBRATION) {
    Serial.println("⚠️ FORCE_CALIBRATION is TRUE — calibrating MQ3 sensors!");
    Serial.println("Please expose to clean air (no alcohol or VOCs)...");

    R0_MQ3_BOTTOM = calibrateSensor(MQ3_BOTTOM_PIN);
    R0_MQ3_TOP = calibrateSensor(MQ3_TOP_PIN);

    Serial.print("✅ Calibration complete. Copy these values:\nR0_MQ3_BOTTOM = ");
    Serial.print(R0_MQ3_BOTTOM);
    Serial.print("; R0_MQ3_TOP = ");
    Serial.println(R0_MQ3_TOP);
  } else {
    Serial.println("ℹ️ Using fixed R0 values — skipping calibration.");
  }

  lastPublishTime = millis();
  startTime = millis();
}

void loop() {
  if (!mqttClient.connected()) reconnectMQTT();
  mqttClient.loop();

  unsigned long currentTime = millis();

  if (currentTime - lastPublishTime >= PUBLISH_INTERVAL) {
    float temp = 0, humidity = 0, pressure = 0, gas = 0;
    if (bmeAvailable && bme.performReading()) {
      temp = bme.temperature;
      humidity = bme.humidity;
      pressure = bme.pressure / 100.0;
      gas = bme.gas_resistance / 1000.0;
    }

    int bottomAnalog = analogRead(MQ3_BOTTOM_PIN);
    int topAnalog = analogRead(MQ3_TOP_PIN);
    float bottomPPM = analogToPPM(bottomAnalog, R0_MQ3_BOTTOM);
    float topPPM = analogToPPM(topAnalog, R0_MQ3_TOP);

    char jsonPayload[512];
    snprintf(jsonPayload, sizeof(jsonPayload),
      "{"
      "\"time\":%lu,"
      "\"BME_Temp\":%.2f,"
      "\"BME_Humidity\":%.2f,"
      "\"BME_VOC_Ohm\":%.2f,"
      "\"MQ3_Bottom_Analog\":%d,"
      "\"MQ3_Bottom_PPM\":%.2f,"
      "\"MQ3_Top_Analog\":%d,"
      "\"MQ3_Top_PPM\":%.2f"
      "}",
      currentTime, temp, humidity, gas,
      bottomAnalog, bottomPPM,
      topAnalog, topPPM
    );

    Serial.println("Publishing to topic:");
    Serial.println(jsonPayload);
    mqttClient.publish(mqtt_topic, jsonPayload);

    lastPublishTime = currentTime;
  }

  if (currentTime - startTime >= RESET_INTERVAL) {
    Serial.println("🔁 Performing scheduled soft reset.");
    delay(1000);
    esp_restart();
  }
}
