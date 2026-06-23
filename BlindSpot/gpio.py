import Jetson.GPIO as GPIO
import time

# Định nghĩa chân cắm (Đang dùng sơ đồ đánh số BOARD - đếm theo thứ tự chân vật lý)
TRIG_PIN = 11 # Chân số 11 trên Header
ECHO_PIN = 12 # Chân số 12 trên Header (NHỚ QUẢN LÝ ĐIỆN ÁP 3.3V CHÂN NÀY!)

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    # Khởi tạo chân TRIG ở mức THẤP
    GPIO.output(TRIG_PIN, GPIO.LOW)
    print("Đang chờ cảm biến ổn định...")
    time.sleep(2)

def get_distance():
    # 1. Phát một xung HIGH dài 10 micro-giây để kích hoạt cảm biến
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    pulse_start = time.time()
    pulse_end = time.time()

    # 2. Đợi chân ECHO lên mức HIGH (Bắt đầu đếm)
    # Thêm timeout để tránh code bị treo vĩnh viễn nếu mất kết nối
    timeout = time.time() + 0.1
    while GPIO.input(ECHO_PIN) == GPIO.LOW and time.time() < timeout:
        pulse_start = time.time()

    # 3. Đợi chân ECHO về mức LOW (Kết thúc đếm)
    timeout = time.time() + 0.1
    while GPIO.input(ECHO_PIN) == GPIO.HIGH and time.time() < timeout:
        pulse_end = time.time()

    # 4. Tính toán khoảng cách
    pulse_duration = pulse_end - pulse_start
    # Tốc độ âm thanh là 34300 cm/s. Chia 2 vì sóng đi và về.
    distance = (pulse_duration * 34300) / 2
    return round(distance, 1)

if __name__ == '__main__':
    try:
        setup()
        print("Bắt đầu đọc khoảng cách. Nhấn Ctrl+C để thoát.")
        while True:
            dist = get_distance()
            print(f"Khoảng cách: {dist} cm")
            time.sleep(0.1) # Đọc 10 lần/giây
            
    except KeyboardInterrupt:
        print("Đã dừng chương trình.")
    finally:
        GPIO.cleanup() # Đưa các chân về trạng thái an toàn