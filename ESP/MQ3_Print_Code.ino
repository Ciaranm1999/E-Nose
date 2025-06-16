// MQ-3 pin configuration
const int MQ3_A_PIN_1 = 35; // Analog output from MQ-3 (first sensor)
const int MQ3_D_PIN_1 = 26; // Digital output from MQ-3 (first sensor)
const int MQ3_A_PIN_2 = 34; // Analog output from MQ-3 (second sensor)
const int MQ3_D_PIN_2 = 25; // Digital output from MQ-3 (second sensor)

// MQ-3 calibration values
float R0_MQ3_1 = 10.0;  // Will be calibrated
float R0_MQ3_2 = 10.0;  // Will be calibrated
const float RL_VALUE = 10.0;  // Load resistance on the board (kOhms)
const float Vc = 5.0;   // Supply voltage (5V for MQ-3)
const float CLEAN_AIR_FACTOR = 60.0; // RS/R0 ratio in clean air for MQ-3

// Calibration parameters
const int CALIBRATION_SAMPLES = 50;    // Number of samples for calibration
const int CALIBRATION_INTERVAL = 500;  // Time between readings (milliseconds)
const int WARMUP_TIME = 30;           // Warmup time in seconds

void setup() {
  Serial.begin(115200);

  pinMode(MQ3_D_PIN_1, INPUT);
  pinMode(MQ3_D_PIN_2, INPUT);
  
  Serial.println("MQ-3 Alcohol Sensor Starting...");
  Serial.println("Allowing sensors to warm up...");
  
  // Simple warmup countdown
  for(int i = WARMUP_TIME; i > 0; i--) {
    Serial.print("Warming up: ");
    Serial.print(i);
    Serial.println(" seconds remaining");
    delay(1000);
  }
  
  Serial.println("\nStarting sensor diagnostics...");
  // Check sensor readings before calibration
  checkSensorReadings();
  
  Serial.println("\nCalibrating sensors in clean air...");
  Serial.println("Please ensure sensors are in clean air with no alcohol present.");
  
  // Calibrate each sensor
  R0_MQ3_1 = calibrateSensor(MQ3_A_PIN_1);
  R0_MQ3_2 = calibrateSensor(MQ3_A_PIN_2);
  
  Serial.print("\nCalibration complete! R0 values: ");
  Serial.print("Sensor 1 (pin 35): "); Serial.print(R0_MQ3_1);
  Serial.print(" | Sensor 2 (pin 34): "); Serial.println(R0_MQ3_2);
}

// Function to check initial sensor readings 
void checkSensorReadings() {
  int val1 = analogRead(MQ3_A_PIN_1);
  int val2 = analogRead(MQ3_A_PIN_2);
  
  Serial.println("Initial sensor readings:");
  Serial.print("Sensor 1 (pin 35): "); Serial.print(val1);
  if(val1 < 100) Serial.println(" <- LOW READING, CHECK CONNECTIONS!");
  else Serial.println(" (OK)");
  
  Serial.print("Sensor 2 (pin 34): "); Serial.print(val2);
  if(val2 < 100) Serial.println(" <- LOW READING, CHECK CONNECTIONS!");
  else Serial.println(" (OK)");
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

// Function to calibrate an MQ-3 sensor - properly using the /60 factor
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

// Function to convert analog value to PPM (using per-sensor R0)
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

void loop() {
  // Read analog values
  int analogValue1 = analogRead(MQ3_A_PIN_1);
  int analogValue2 = analogRead(MQ3_A_PIN_2);

  // Calculate PPM values using sensor-specific calibration values
  float ppm1 = analogToPPM(analogValue1, R0_MQ3_1);
  float ppm2 = analogToPPM(analogValue2, R0_MQ3_2);
  
  // Read digital values
  int digitalValue1 = digitalRead(MQ3_D_PIN_1);
  int digitalValue2 = digitalRead(MQ3_D_PIN_2);

  // Display results for sensor 1
  Serial.print("Sensor 1 (pin 35) Analog: ");
  Serial.print(analogValue1);
  Serial.print("  |  Digital: ");
  Serial.print(digitalValue1 == HIGH ? "No Gas" : "Gas Detected");
  Serial.print("  |  PPM: ");
  Serial.println(ppm1);

  // Display results for sensor 2
  Serial.print("Sensor 2 (pin 34) Analog: ");
  Serial.print(analogValue2);
  Serial.print("  |  Digital: ");
  Serial.print(digitalValue2 == HIGH ? "No Gas" : "Gas Detected");
  Serial.print("  |  PPM: ");
  Serial.println(ppm2);

  Serial.println("-----------------------------");
  delay(1000); // Wait 1 second
}