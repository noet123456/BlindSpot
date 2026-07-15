import cv2
import numpy as np
import serial
import threading
from ultralytics import YOLO

# ==========================================
# 1. ĐỌC TÍN HIỆU SERIAL TỪ CỔNG COM
# ==========================================
current_distance = -1.0

def read_serial_data(port='/dev/ttyUSB0', baudrate=9600):
    global current_distance
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[HỆ THỐNG] Đã kết nối Arduino tại {port}")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.replace('.','',1).isdigit(): # Cho phép số thập phân
                    current_distance = float(line)
    except Exception as e:
        print(f"[LỖI] Không thể kết nối Serial: {e}")

serial_thread = threading.Thread(target=read_serial_data, daemon=True)
serial_thread.start()

# ==========================================
# 2. KHỞI TẠO CAMERA VÀ MODEL OBB
# ==========================================
# Khai báo mô hình YOLO OBB (Ví dụ: yolo11n-obb.engine)
model = YOLO('yolo11n-obb.pt', task='obb') 

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ==========================================
# 3. GIAI ĐOẠN CHỜ POPUP BIẾN MẤT
# ==========================================
print("[HỆ THỐNG] Đang chờ Đầu ghi XVR ẩn khung thông báo...")
clear_frames_count = 0 

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Lấy mẫu một ô vuông 50x50 pixel ở tâm màn hình
    roi = frame[mid_h - 25 : mid_h + 25, mid_w - 25 : mid_w + 25]
    
    # Chuyển sang ảnh xám và tính giá trị độ sáng trung bình (0 -> 255)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_roi)

    # Nếu bảng thông báo màu trắng, độ sáng trung bình sẽ rất cao (ví dụ > 180)
    # Ngược lại, nếu nền bình thường, độ sáng sẽ thấp hơn.
    if avg_brightness < 180:
        clear_frames_count += 1
    else:
        clear_frames_count = 0 # Nạp lại bộ đếm nếu vẫn thấy màu trắng

    # Hiển thị vùng lấy mẫu để bạn dễ debug (Vẽ viền đỏ)
    cv2.rectangle(frame, (mid_w - 25, mid_h - 25), (mid_w + 25, mid_h + 25), (0, 0, 255), 2)
    cv2.putText(frame, f"Checking Popup... Brightness: {avg_brightness:.0f}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Startup Waiting", frame)

    # Đợi ổn định 10 khung hình liên tiếp không có màu trắng thì thoát vòng lặp chờ
    if clear_frames_count > 10:
        print("[HỆ THỐNG] Màn hình đã sạch! Bắt đầu kích hoạt AI...")
        cv2.destroyWindow("Startup Waiting")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

# ==========================================
# 4. GIAI ĐOẠN AI BATCH INFERENCE & SENSOR FUSION OBB
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Cắt màn hình
    cam1 = frame[0:mid_h, 0:mid_w]           
    cam2 = frame[0:mid_h, mid_w:w]           
    cam3 = frame[mid_h:h, 0:mid_w]           
    cam4 = frame[mid_h:h, mid_w:w]           

    batch_frames = [cam1, cam2, cam3, cam4]
    results = list(model(batch_frames, stream=True, conf=0.5, verbose=False))

    # Biến theo dõi vật thể có OBB lớn nhất TOÀN HỆ THỐNG
    global_max_area = 0
    target_cam_idx = -1
    target_obb_corners = None
    target_class_id = -1

    # 4.1. Thuật toán tìm OBB lớn nhất
    for i, result in enumerate(results):
        # Lưu ý: Mô hình OBB sử dụng result.obb thay vì result.boxes
        if result.obb is not None:
            for obb in result.obb:
                # Trích xuất (x, y, width, height, rotation)
                _, _, bw, bh, _ = obb.xywhr[0].cpu().numpy()
                area = bw * bh 
                
                if area > global_max_area:
                    global_max_area = area
                    target_cam_idx = i
                    target_class_id = int(obb.cls[0].item())
                    # Lấy tọa độ 4 góc của hình chữ nhật nghiêng (shape: 4x2)
                    target_obb_corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)

    # 4.2. Vẽ và Gán thông số lên Camera chứa vật thể lớn nhất
    for i, result in enumerate(results):
        # Plot mặc định của YOLO sẽ tự động vẽ các khung OBB lên ảnh
        annotated_frame = result.plot()
        
        cv2.putText(annotated_frame, f"CAM {i+1}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Nếu đây là luồng Camera chứa vật thể to nhất, tiến hành gán cự ly
        if i == target_cam_idx and target_obb_corners is not None and current_distance > 0:
            
            # Tìm điểm đỉnh cao nhất (Y nhỏ nhất) trong 4 góc OBB để đặt Text không bị che lấp
            top_point = target_obb_corners[np.argmin(target_obb_corners[:, 1])]
            x_text, y_text = top_point[0], top_point[1] - 10
            
            # Vẽ nền chữ nhật cho chữ dễ đọc
            text = f"{current_distance} cm"
            cv2.putText(annotated_frame, text, (x_text, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            class_name = model.names[target_class_id]
            print(f"[CẢNH BÁO] {class_name.upper()} tại CAM {i+1} | Diện tích: {global_max_area:.0f} | Khoảng cách: {current_distance} cm")

        cv2.imshow(f"Luong CAM {i+1}", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()