#include <SoftwareSerial.h>

// Định nghĩa 4 cổng Serial ảo tương ứng với 4 cảm biến (Cú pháp: SoftwareSerial(rxPin, txPin))
SoftwareSerial sensorF(2, 3); // Cảm biến trước (Front)
SoftwareSerial sensorB(4, 5); // Cảm biến sau (Back)
SoftwareSerial sensorL(6, 7); // Cảm biến trái (Left)
SoftwareSerial sensorR(8, 9); // Cảm biến phải (Right)

float readSensor(SoftwareSerial &sensor) {
  sensor.listen(); // Kích hoạt cổng Serial của cảm biến này để nghe
  delay(10);       // Đợi cổng ổn định chuyển mạch
  
  // Gửi lệnh kích hoạt 0x55 xuống cảm biến (Yêu cầu cảm biến đo ngay lập tức)
  sensor.write(0x55); 
  
  unsigned long startTime = millis();
  unsigned char data[4] = {0};
  int byteCount = 0;
  
  // Chờ đọc đủ 4 byte dữ liệu trả về với Timeout là 50ms
  while (millis() - startTime < 50) {
    if (sensor.available() > 0) {
      data[byteCount] = sensor.read();
      
      // Đồng bộ hóa: Byte đầu tiên bắt buộc phải là 0xFF
      if (byteCount == 0 && data[0] != 0xFF) {
        continue; 
      }
      byteCount++;
      if (byteCount == 4) break;
    }
  }
  
  // Nếu nhận đủ 4 byte, kiểm tra tính đúng đắn của dữ liệu (Checksum)
  if (byteCount == 4) {
    int sum = (data[0] + data[1] + data[2]) & 0x00FF;
    if (sum == data[3]) {
      float distance_mm = (data[1] << 8) + data[2];
      return distance_mm / 10.0; // Đổi từ mm sang cm
    }
  }
  return 0.0; // Trả về -1.0 nếu cảm biến bị lỗi hoặc không có vật cản
}

void setup() {
  Serial.begin(9600); // Giao tiếp với Jetson Orin Nano
  sensorF.begin(9600);
  sensorB.begin(9600);
  sensorL.begin(9600);
  sensorR.begin(9600);
}

void loop() {
  // Đọc tuần tự từng cảm biến
  float f = readSensor(sensorF);
  float b = readSensor(sensorB);
  float l = readSensor(sensorL);
  float r = readSensor(sensorR);
  
  // Xuất dữ liệu lên Jetson theo định dạng: x y z t (ngăn cách bằng khoảng trắng)
  Serial.print(f, 1);
  Serial.print(" ");
  Serial.print(b, 1);
  Serial.print(" ");
  Serial.print(l, 1);
  Serial.print(" ");
  Serial.println(r, 1); // Dùng println để báo kết thúc dòng dữ liệu
  
  delay(50); // Chu kỳ quét toàn hệ thống là 20Hz (20 lần/giây)
}