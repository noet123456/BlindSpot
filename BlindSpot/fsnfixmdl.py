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
                if line.replace('.','',1).isdigit():
                    current_distance = float(line)
    except Exception as e:
        print(f"[LỖI] Không thể kết nối Serial: {e}")

serial_thread = threading.Thread(target=read_serial_data, daemon=True)
serial_thread.start()

# ==========================================
# 2. KHỞI TẠO CAMERA VÀ MODEL SEGMENTATION
# ==========================================
# Đổi sang mô hình Segment (ví dụ: yolo11n-seg.pt)
model = YOLO('yolov8n-seg.pt') 

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

    roi = frame[mid_h - 25 : mid_h + 25, mid_w - 25 : mid_w + 25]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_roi)

    if avg_brightness < 180:
        clear_frames_count += 1
    else:
        clear_frames_count = 0 

    cv2.rectangle(frame, (mid_w - 25, mid_h - 25), (mid_w + 25, mid_h + 25), (0, 0, 255), 2)
    cv2.putText(frame, f"Checking Popup... Brightness: {avg_brightness:.0f}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Startup Waiting", frame)

    if clear_frames_count > 10:
        print("[HỆ THỐNG] Màn hình đã sạch! Bắt đầu kích hoạt AI...")
        cv2.destroyWindow("Startup Waiting")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

# ==========================================
# 4. GIAI ĐOẠN AI BATCH INFERENCE & SENSOR FUSION SEGMENT
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    cam1 = frame[0:mid_h, 0:mid_w]           
    cam2 = frame[0:mid_h, mid_w:w]           
    cam3 = frame[mid_h:h, 0:mid_w]           
    cam4 = frame[mid_h:h, mid_w:w]           

    batch_frames = [cam1, cam2, cam3, cam4]
    results = list(model(batch_frames, stream=True, conf=0.5, verbose=False))

    global_max_area = 0
    target_cam_idx = -1
    target_rect_corners = None
    target_class_id = -1

    # 4.1. Thuật toán tìm Vật thể có diện tích lớn nhất bằng Segment + minAreaRect
    for i, result in enumerate(results):
        # Kiểm tra xem có mask (phân mảnh) nào được phát hiện không
        if result.masks is not None:
            # Lặp qua từng mảng tọa độ viền của vật thể
            for idx, segment_points in enumerate(result.masks.xy):
                if len(segment_points) == 0:
                    continue
                
                # Ép mảng tọa độ sang định dạng NumPy float32
                contour = np.array(segment_points, dtype=np.float32)
                
                # Tính hình chữ nhật xoay nhỏ nhất bọc ngoài các điểm pixel
                rect = cv2.minAreaRect(contour)
                (cx, cy), (bw, bh), angle = rect
                area = bw * bh 
                
                if area > global_max_area:
                    global_max_area = area
                    target_cam_idx = i
                    target_class_id = int(result.boxes.cls[idx].item())
                    
                    # Chuyển đổi thông số rect thành tọa độ 4 góc nguyên để vẽ
                    box = cv2.boxPoints(rect)
                    target_rect_corners = np.int0(box)

    # 4.2. Vẽ và Gán thông số lên Camera chứa vật thể lớn nhất
    for i, result in enumerate(results):
        # Plot mặc định của YOLO sẽ tự động vẽ lớp phủ Segment (Mask) lên ảnh
        annotated_frame = result.plot()
        
        cv2.putText(annotated_frame, f"CAM {i+1}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Gán thông số nếu đây là luồng chứa vật thể to nhất và cảm biến có dữ liệu
        if i == target_cam_idx and target_rect_corners is not None and current_distance > 0:
            
            # Chủ động vẽ thêm đường viền hình chữ nhật bao quanh vùng Segment
            cv2.drawContours(annotated_frame, [target_rect_corners], 0, (0, 255, 0), 2)
            
            # Tìm điểm đỉnh cao nhất để đặt Text
            top_point = target_rect_corners[np.argmin(target_rect_corners[:, 1])]
            x_text, y_text = top_point[0], top_point[1] - 10
            
            text = f"{current_distance} cm"
            cv2.putText(annotated_frame, text, (x_text, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            class_name = model.names[target_class_id]
            print(f"[CẢNH BÁO] {class_name.upper()} tại CAM {i+1} | Diện tích: {global_max_area:.0f} px^2 | Khoảng cách: {current_distance} cm")

        cv2.imshow(f"Luong CAM {i+1}", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()