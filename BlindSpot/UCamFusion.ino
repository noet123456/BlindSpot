#include <SoftwareSerial.h>

// ==========================================
// KHAI BÁO CẢM BIẾN 1: JSN-SR04T (UART)
// ==========================================
#define UART_RX_PIN_B 2  // Nối với chân ECHO (TX của cảm biến)
#define UART_TX_PIN_B 3  // Nối với chân TRIG (RX của cảm biến)
SoftwareSerial backSensor(UART_RX_PIN_B, UART_TX_PIN_B);

// ==========================================
// KHAI BÁO CẢM BIẾN 2, 3, 4: HC-SR04 (TRIG/ECHO)
// ==========================================
#define TRIG_F 4
#define ECHO_F 5

#define TRIG_R 6
#define ECHO_R 7

#define TRIG_L 8
#define ECHO_L 9

// Ngưỡng Timeout 30000 micro-giây (~ 5 mét). Quá thời gian này coi như không có vật cản.
#define TIMEOUT_US 30000 

void setup() {
  Serial.begin(9600);      // Giao tiếp Jetson
  backSensor.begin(9600);  // Giao tiếp module UART
  
  // Khởi tạo In/Out cho 3 module Trig/Echo
  pinMode(TRIG_F, OUTPUT); pinMode(ECHO_F, INPUT);
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);
}

// Hàm đọc module UART (Giả định đang dùng chế độ Đợi lệnh 0x55)
float readUARTSensor() {
  backSensor.write(0x55); 
  unsigned long startTime = millis();
  unsigned char data[4] = {0};
  int byteCount = 0;
  
  while (millis() - startTime < 50) {
    if (backSensor.available() > 0) {
      data[byteCount] = backSensor.read();
      if (byteCount == 0 && data[0] != 0xFF) continue; 
      byteCount++;
      if (byteCount == 4) break;
    }
  }
  
  if (byteCount == 4) {
    int sum = (data[0] + data[1] + data[2]) & 0x00FF;
    if (sum == data[3]) {
      return ((data[1] << 8) + data[2]) / 10.0;
    }
  }
  return 0.0;
}

// Hàm đọc module Trig/Echo
float readTrigEchoSensor(int trigPin, int echoPin) {
  // Dọn sạch nhiễu trên chân phát
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  // Phát xung 10us
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Đọc xung trả về với Timeout bảo vệ hệ thống
  long duration = pulseIn(echoPin, HIGH, TIMEOUT_US);
  
  if (duration == 0) {
    return 0.0; // Báo -1.0 nếu vượt quá tầm đo hoặc mất kết nối
  }
  
  // Tính khoảng cách (cm)
  return (duration * 0.034) / 2.0;
}

void loop() {
  // 1. Quét lần lượt 4 cảm biến
  float B = readUARTSensor();
  float F = readTrigEchoSensor(TRIG_F, ECHO_F);
  float L = readTrigEchoSensor(TRIG_L, ECHO_L);
  float R = readTrigEchoSensor(TRIG_R, ECHO_R);
  
  // 2. Xuất dữ liệu lên cổng USB theo định dạng: x y z t
  Serial.print(F, 1); Serial.print(" ");
  Serial.print(B, 1); Serial.print(" ");
  Serial.print(L, 1); Serial.print(" ");
  Serial.println(R, 1); 
  
  // Nghỉ 50ms giữa các vòng lặp để tránh dội sóng âm (Echo Interference)
  delay(50); 
}