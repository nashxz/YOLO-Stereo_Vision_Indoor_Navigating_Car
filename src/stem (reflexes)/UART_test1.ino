// Define your new custom UART pins
#define RXD2 32  // Connect to Jetson Pin 8 (TX)
#define TXD2 33  // Connect to Jetson Pin 10 (RX)

void setup() {
  // Laptop/USB-C Monitor
  Serial.begin(115200); 
  
  // Jetson Nano Monitor on Pins 32 and 33
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);
  
  delay(1000);
  Serial.println("System Online. Monitoring Jetson on GPIO 32/33...");
}

void loop() {
  if (Serial2.available() > 0) {
    // Read the message from Jetson
    String message = Serial2.readStringUntil('\n');
    
    // Print to your laptop Serial Monitor
    Serial.print("JETSON DATA: ");
    Serial.println(message);
  }
}