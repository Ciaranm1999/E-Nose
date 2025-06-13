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

// MQ-3 pin configuration - Bottom is pin 35, Top is pin 34
const int MQ3_BOTTOM_PIN = 35;
const int MQ3_BOTTOM_D_PIN = 26;
const int MQ3_TOP_PIN = 34;
const int MQ3_TOP_D_PIN = 25;

// For backward compatibility with existing array code
const int MQ3_A_PINS[3] = {32, MQ3_BOTTOM_PIN, MQ3_TOP_PIN};
const int MQ3_D_PINS[3] = {25, MQ3_BOTTOM_D_PIN, MQ3_TOP_D_PIN};

// MQ-3 calibration values
float R0_MQ3_BOTTOM = 10.0;  // Will be calibrated
float R0_MQ3_TOP = 10.0;     // Will be calibrated
const float RL_VALUE = 10.0;  // Load resistance on the board (kOhms)
const float Vc = 5.0;         // Supply voltage (5V for MQ-3)
const float CLEAN_AIR_FACTOR = 60.0; // RS/R0 ratio in clean air for MQ-3

// Calibration parameters
const int CALIBRATION_SAMPLES = 20;    // Reduced from 50 to speed up startup
const int CALIBRATION_INTERVAL = 250;  // Reduced from 500 to speed up startup

const int LED_PIN = 2;
bool bmeAvailable = false;

const unsigned long PUBLISH_INTERVAL = 120000; // 10 seconds
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

// Function to calculate sensor resistance
float readSensorResistance(int sensorPin) {
  int analogValue = analogRead(sensorPin);
  
  // Protection against zero readings
  if(analogValue < 1) analogValue = 1;
  
  float sensorVoltage = analogValue * (Vc / 4095.0);
  
  // Protection against zero voltage
  if(sensorVoltage < 0.1) sensorVoltage = 0.1;
  
  float rs = ((Vc - sensorVoltage) / sensorVoltage) * RL_VALUE;
  return rs;
}

// Function to calibrate an MQ-3 sensor
float calibrateSensor(int sensorPin) {
  float rs_sum = 0.0;
  int valid_samples = 0;
  
  // Take multiple readings and average them
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    float rs = readSensorResistance(sensorPin);
    
    // Only use reasonable readings for calibration
    if(rs > 0.1 && rs < 1000) {  // Sanity check for reasonable resistance values
      rs_sum += rs;
      valid_samples++;
    }
    
    delay(CALIBRATION_INTERVAL);
  }
  
  // Check if we have enough valid samples
  if(valid_samples < CALIBRATION_SAMPLES/2) {
    Serial.print("WARNING: Sensor on pin ");
    Serial.print(sensorPin);
    Serial.println(" has too many invalid readings. Using default R0.");
    return 10.0; // Default value
  }
  
  // Calculate average resistance in clean air
  float rs_avg = rs_sum / valid_samples;
  
  // For MQ-3, R0 = rs_clean_air/60 (based on datasheet)
  float r0 = rs_avg / CLEAN_AIR_FACTOR;
  
  // Sanity check on R0
  if(r0 < 0.1 || r0 > 100) {
    Serial.print("WARNING: Calibration value out of range on pin ");
    Serial.println(sensorPin);
    return 10.0; // Default value
  }
  
  return r0;
}

// Function to convert analog value to PPM
float analogToPPM(int analogValue, float r0) {
  // Protection against zero readings
  if(analogValue < 1) analogValue = 1;
  
  // Convert analog reading to sensor resistance
  float sensorVoltage = analogValue * (Vc / 4095.0);
  if(sensorVoltage < 0.1) sensorVoltage = 0.1; // Protection against zero voltage
  
  float rs = ((Vc - sensorVoltage) / sensorVoltage) * RL_VALUE;
  
  // Convert to ratio
  float ratio = rs / r0;
  
  // Use power regression formula to calculate PPM
  // PPM = a * (rs/r0)^b where a=607.9 and b=-2.868 for MQ-3 (approximate values)
  float ppm = 607.9 * pow(ratio, -2.868);
  
  return ppm;
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
  pinMode(MQ3_BOTTOM_D_PIN, INPUT);
  pinMode(MQ3_TOP_D_PIN, INPUT);

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
  // Try the first address (0x76)
  if (bme.begin(0x76)) {
    bmeAvailable = true;
    Serial.println("✅ BME680 detected at address 0x76!");
  }
  // If that fails, try the second address (0x77)
  else if (bme.begin(0x77)) {
    bmeAvailable = true;
    Serial.println("✅ BME680 detected at address 0x77!");
  }
  // If both fail, report failure
  else {
    Serial.println("❌ BME680 not found at either address. Continuing without it...");
    bmeAvailable = false;
  }
  
  // Configure BME680 if available
  if (bmeAvailable) {
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setGasHeater(320, 150);
  }

  analogReadResolution(12);
  for (int i = 0; i < 3; i++) {
    analogSetPinAttenuation(MQ3_A_PINS[i], ADC_11db);
    pinMode(MQ3_D_PINS[i], INPUT);
  }
  
  // Calibrate MQ3 sensors
  Serial.println("\nCalibrating MQ3 sensors in clean air...");
  Serial.println("Please ensure sensors are in clean air with no alcohol present.");
  
  R0_MQ3_BOTTOM = calibrateSensor(MQ3_BOTTOM_PIN);
  R0_MQ3_TOP = calibrateSensor(MQ3_TOP_PIN);
  
  Serial.print("\nCalibration complete! R0 values: ");
  Serial.print("Bottom MQ3 (pin 35): "); Serial.print(R0_MQ3_BOTTOM);
  Serial.print(" | Top MQ3 (pin 34): "); Serial.println(R0_MQ3_TOP);

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
    
    // Read MQ3 sensors
    int bottomAnalog = analogRead(MQ3_BOTTOM_PIN);
    int topAnalog = analogRead(MQ3_TOP_PIN);
    int bottomDigital = digitalRead(MQ3_BOTTOM_D_PIN);
    int topDigital = digitalRead(MQ3_TOP_D_PIN);
    
    // Calculate PPM values
    float bottomPPM = analogToPPM(bottomAnalog, R0_MQ3_BOTTOM);
    float topPPM = analogToPPM(topAnalog, R0_MQ3_TOP);

    char jsonPayload[512];
    
    // Format with the requested field names
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
      currentTime,
      temp, humidity, gas,
      bottomAnalog, bottomPPM,
      topAnalog, topPPM
    );

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