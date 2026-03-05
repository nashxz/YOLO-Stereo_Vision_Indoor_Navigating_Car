// ===== PIN DEFINITIONS =====
#define AIN1  23
#define AIN2  22
#define PWMA  21

#define BIN1  19
#define BIN2  18
#define PWMB  17

#define STBY  16

// ===== MOTOR CONTROL =====
void motors_stop() {
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);

  digitalWrite(PWMA, LOW);
  digitalWrite(PWMB, LOW);
}

void motors_forward_full() {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);

  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);

  digitalWrite(PWMA, HIGH);   // FULL voltage (no PWM)
  digitalWrite(PWMB, HIGH);
}

void setup() {
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(PWMB, OUTPUT);
  pinMode(STBY, OUTPUT);

  digitalWrite(STBY, HIGH);  // enable driver
}

void loop() {
  motors_forward_full();
  delay(5000);

  motors_stop();
  delay(5000);
}