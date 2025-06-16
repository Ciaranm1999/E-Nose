#include <DHT.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <PID_v1.h>
#include <avr/wdt.h>  // Watchdog timer

// === CONFIG FLAGS ===
const bool FAN_ALWAYS_ON = true;  // true = fan runs constantly, false = burst mode
const int FAN_PWM_VALUE = 255;  // Set between 0 (off) and 255 (full speed)
#define USE_OLED true
#define SHOW_INDIVIDUAL_TEMPS true

// === Pins ===
#define DHTPIN1 2
#define DHTPIN2 3
#define DHTTYPE DHT11
#define FAN_PIN 6
#define PELTIER_PIN 7
#define HUMIDIFIER_CTRL_PIN 4 

// === OLED ===
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// === Sensors ===
DHT dht1(DHTPIN1, DHTTYPE);
DHT dht2(DHTPIN2, DHTTYPE);

// === Setpoints ===
double Setpoint = 32.0;         // Temperature °C
double HumiditySetpoint = 80.0; // Humidity %

#define MIN_COOL_PWM 245

double Input;   // Average temp
double Output;
double Kp = 60, Ki = 0.5, Kd = 20;  // Adjusted tuning values
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

// === Humidifier control ===
bool humidifierIsOn = false;
unsigned long lastBurstTime = 0;
bool waitingForNextBurst = false;

// === Fan burst control ===
const unsigned long fanBurstDuration = 5000;   // Fan ON for -- seconds
const unsigned long fanRestDuration = 30000;   // Wait -- seconds between bursts
unsigned long lastFanBurstTime = 0;
bool fanIsOn = false;

// === Reset Timer ===
unsigned long lastResetTime = 0;
const unsigned long resetInterval = 1800000;  // 30 minutes in milliseconds

void sendPulse() {
  digitalWrite(HUMIDIFIER_CTRL_PIN, HIGH);
  delay(500);
  digitalWrite(HUMIDIFIER_CTRL_PIN, LOW);
  delay(500);
}

void turnHumidifierOn() {
  if (!humidifierIsOn) {
    sendPulse(); // OFF → ON
    humidifierIsOn = true;
  }
}

void turnHumidifierOff() {
  if (humidifierIsOn) {
    sendPulse(); // ON → Interval
    sendPulse(); // Interval → OFF
    humidifierIsOn = false;
  }
}

void burstHumidifier(int durationMs) {
  turnHumidifierOn();
  delay(durationMs);
  turnHumidifierOff();
}

void softwareReset() {
  wdt_enable(WDTO_15MS);  // Enable watchdog timer with shortest timeout
  while (1);              // Wait for watchdog to trigger reset
}

void setup() {
  Serial.begin(9600);
  dht1.begin();
  dht2.begin();

  pinMode(FAN_PIN, OUTPUT);
  pinMode(PELTIER_PIN, OUTPUT);
  pinMode(HUMIDIFIER_CTRL_PIN, OUTPUT);
  digitalWrite(HUMIDIFIER_CTRL_PIN, LOW);
  digitalWrite(FAN_PIN, LOW);  // Start fan OFF

  myPID.SetMode(AUTOMATIC);
  myPID.SetOutputLimits(0, 255);

  lastResetTime = millis();

#if USE_OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found!");
  } else {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("OLED Initialized");
    display.display();
    delay(1000);
  }
#endif
}

void loop() {
  unsigned long currentMillis = millis();

  // === RESET CHECK ===
  if (currentMillis - lastResetTime >= resetInterval) {
    Serial.println("Resetting Arduino...");
    delay(1000);  // Give time for serial message to be sent
    softwareReset();  // Trigger watchdog reset
  }

  float t1 = dht1.readTemperature();
  float h1 = dht1.readHumidity();
  float t2 = dht2.readTemperature();
  float h2 = dht2.readHumidity();

  if (isnan(t1) || isnan(h1) || isnan(t2) || isnan(h2)) {
    Serial.println("Sensor read failed!");
    return;
  }

  Input = (t1 + t2) / 2.0;
  double avgHum = (h1 + h2) / 2.0;

  // === PID TEMP CONTROL ===
  int minPWM = 0;

  if (Setpoint <= 24) {
    if (Input >= 24.0) {
      minPWM = MIN_COOL_PWM;
    } else if ((Input - Setpoint) > 0.25) {
      minPWM = MIN_COOL_PWM;
    } else {
      minPWM = 0;
    }
  } else {
    if (Setpoint <= 26) minPWM = MIN_COOL_PWM;
    else if (Setpoint <= 28) minPWM = 120;
    else if (Setpoint <= 30) minPWM = 190;
    else if (Setpoint <= 32) minPWM = 220;
    else minPWM = 255;
  }

  // Clamp: Full power until close to setpoint
  if (Setpoint - Input > 0.25) {
    Output = 255;
  } else {
    myPID.Compute();
  }

  // Enforce minimum cooling threshold if applicable
  if (Input > Setpoint && Output < minPWM) {
    Output = minPWM;
  }

  analogWrite(PELTIER_PIN, (int)Output);


  // === FAN CONTROL ===
    if (FAN_ALWAYS_ON) {
    analogWrite(FAN_PIN, FAN_PWM_VALUE);  // Fan always ON with set PWM
  } else {
    if (!fanIsOn && (currentMillis - lastFanBurstTime >= fanRestDuration)) {
      analogWrite(FAN_PIN, FAN_PWM_VALUE);  // Turn fan ON with set PWM
      fanIsOn = true;
      lastFanBurstTime = currentMillis;
    }

    if (fanIsOn && (currentMillis - lastFanBurstTime >= fanBurstDuration)) {
      analogWrite(FAN_PIN, 0);   // Turn fan OFF
      fanIsOn = false;
    }
  }


  // === HUMIDITY BURST CONTROL ===
  double humidityError = HumiditySetpoint - avgHum;

  if (!waitingForNextBurst) {
    if (humidityError > 5.0) {
      burstHumidifier(2000);
      lastBurstTime = currentMillis;
      waitingForNextBurst = true;
    } else if (humidityError > 0.5) {
      burstHumidifier(1000);
      lastBurstTime = currentMillis;
      waitingForNextBurst = true;
    } else {
      turnHumidifierOff();
    }
  }

  if (waitingForNextBurst && (currentMillis - lastBurstTime >= 15000)) {
    waitingForNextBurst = false;
  }

  // === SERIAL OUTPUT ===
#if SHOW_INDIVIDUAL_TEMPS
  Serial.print("T1: "); Serial.print(t1, 1); Serial.print("°C | ");
  Serial.print("H1: "); Serial.print(h1, 1); Serial.print("% || ");
  Serial.print("T2: "); Serial.print(t2, 1); Serial.print("°C | ");
  Serial.print("H2: "); Serial.print(h2, 1); Serial.print("% || ");
#else
  Serial.print("Avg Temp: ");
  Serial.print(Input, 1);
  Serial.print(" °C | Avg Hum: ");
  Serial.print(avgHum, 1);
#endif

  Serial.print(" % | ");
  Serial.print("Setpoint T: ");
  Serial.print(Setpoint, 1);
  Serial.print(" °C | Setpoint H: ");
  Serial.print(HumiditySetpoint, 1);
  Serial.print(" % | ");
  Serial.print("PWM: ");
  Serial.println((int)Output);

  // === OLED OUTPUT ===
#if USE_OLED
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(0, 0);

#if SHOW_INDIVIDUAL_TEMPS
  display.print(t1, 1); display.print("/"); display.println(t2, 1);
  display.setCursor(0, 22);
  display.print(h1, 1); display.print("/"); display.println(h2, 1);
#else
  display.print(Input, 1); display.println("C");
  display.setCursor(0, 20);
  display.print(avgHum, 1); display.println("%");
#endif

  display.setCursor(0, 40);
  display.print("PWM: ");
  display.println((int)Output);
  display.display();
#endif

  delay(2000);
}
