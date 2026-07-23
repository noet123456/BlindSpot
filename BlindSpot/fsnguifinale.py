import cv2
import numpy as np
import serial
import threading
from ultralytics import YOLO

# ==========================================
# 1. ĐỌC TÍN HIỆU SERIAL TỪ 2 CỔNG COM
# ==========================================
# Khởi tạo các biến toàn cục lưu trữ khoảng cách
dist_x = dist_y = dist_z = -1.0
dist_t = -1.0

def read_serial_xyz(port='/dev/ttyUSB0', baudrate=9600):
    global dist_x, dist_y, dist_z
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[HỆ THỐNG] Cổng 1 ({port}) đã kết nối -> Đọc tín hiệu X, Y, Z")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data = line.split() 
                if len(data) == 3:
                    try:
                        dist_x = float(data[0])
                        dist_y = float(data[1])
                        dist_z = float(data[2])
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[LỖI] Không thể kết nối Cổng 1 ({port}): {e}")

def read_serial_t(port='/dev/ttyUSB1', baudrate=9600):
    global dist_t
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[HỆ THỐNG] Cổng 2 ({port}) đã kết nối -> Đọc tín hiệu T")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                try:
                    dist_t = float(line) 
                except ValueError:
                    pass
    except Exception as e:
        print(f"[LỖI] Không thể kết nối Cổng 2 ({port}): {e}")

thread_xyz = threading.Thread(target=read_serial_xyz, args=('/dev/ttyUSB0', 9600), daemon=True)
thread_t = threading.Thread(target=read_serial_t, args=('/dev/ttyUSB1', 9600), daemon=True)
thread_xyz.start()
thread_t.start()


# ==========================================
# 2. KHỞI TẠO CAMERA VÀ MODEL SEGMENTATION
# ==========================================
model = YOLO('yolov8n-seg.pt') # Đảm bảo đúng đường dẫn file model của bạn

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
# 4. GIAI ĐOẠN AI SEGMENT & SENSOR FUSION ĐA LUỒNG
# ==========================================
WINDOW_NAME = "Jetson Orin - Smart Surveillance Dashboard"
cv2.namedWindow(WINDOW_NAME)
view_mode = -1 # -1: Chế độ lưới 4 cam | 0, 1, 2, 3: Phóng to Cam 1, 2, 3, 4

def mouse_click(event, x, y, flags, param):
    global view_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 120 and 20 <= y <= 60:
            view_mode = -1
        if view_mode == -1 and y > 80:
            mid_x, mid_y = 1280 // 2, 720 // 2
            if x < mid_x and y < mid_y: view_mode = 0      
            elif x >= mid_x and y < mid_y: view_mode = 1   
            elif x < mid_x and y >= mid_y: view_mode = 2   
            elif x >= mid_x and y >= mid_y: view_mode = 3  
        if view_mode != -1:
            if 20 <= x <= 80 and 310 <= y <= 410:          
                view_mode = (view_mode - 1) % 4
            elif 1200 <= x <= 1260 and 310 <= y <= 410:    
                view_mode = (view_mode + 1) % 4
cv2.setMouseCallback(WINDOW_NAME, mouse_click)

def draw_button(img, text, pt1, pt2, bg_color=(50, 50, 50)):
    cv2.rectangle(img, pt1, pt2, bg_color, -1)
    cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    txt_x = pt1[0] + (pt2[0] - pt1[0] - text_size[0]) // 2
    txt_y = pt1[1] + (pt2[1] - pt1[1] + text_size[1]) // 2
    cv2.putText(img, text, (txt_x, txt_y), font, 0.8, (255, 255, 255), 2)

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Tách 4 khung hình
    cam1 = frame[0:mid_h, 0:mid_w]           
    cam2 = frame[0:mid_h, mid_w:w]           
    cam3 = frame[mid_h:h, 0:mid_w]           
    cam4 = frame[mid_h:h, mid_w:w]           

    batch_frames = [cam1, cam2, cam3, cam4]
    
    # Ép chạy vòng lặp để tránh lỗi max model size (AssertionError)
    results = []
    for cam in batch_frames:
        res = model(cam, conf=0.5, verbose=False)[0]
        results.append(res)

    global_max_area = 0
    target_cam_idx = -1
    target_rect_corners = None
    target_class_id = -1

    # Tìm vật thể Segment lớn nhất trên toàn bộ 4 Camera
    for i, result in enumerate(results):
        if result.masks is not None:
            for idx, segment_points in enumerate(result.masks.xy):
                if len(segment_points) == 0:
                    continue
                
                contour = np.array(segment_points, dtype=np.float32)
                rect = cv2.minAreaRect(contour)
                (cx, cy), (bw, bh), angle = rect
                area = bw * bh 
                
                if area > global_max_area:
                    global_max_area = area
                    target_cam_idx = i
                    target_class_id = int(result.boxes.cls[idx].item())
                    box = cv2.boxPoints(rect)
                    target_rect_corners = np.int0(box)

    annotated_frames = []
    # Bản đồ ánh xạ biến khoảng cách: [Cam 1, Cam 2, Cam 3, Cam 4] -> [x, y, z, t]
    sensor_values = [dist_x, dist_y, dist_z, dist_t]

    for i, result in enumerate(results):
        ann_frame = result.plot()
        cv2.putText(ann_frame, f"CAM {i+1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cam_distance = sensor_values[i] # Lấy biến khoảng cách tương ứng với Camera hiện tại

        # Xử lý hiển thị thông số lên khung hình có vật thể lớn nhất
        if i == target_cam_idx and target_rect_corners is not None and cam_distance > 0:
            cv2.drawContours(ann_frame, [target_rect_corners], 0, (0, 255, 0), 2)
            top_point = target_rect_corners[np.argmin(target_rect_corners[:, 1])]
            
            cv2.putText(ann_frame, f"{cam_distance} cm", (top_point[0], top_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            class_name = model.names[target_class_id]
            print(f"[CẢNH BÁO] {class_name.upper()} tại CAM {i+1} | Area: {global_max_area:.0f} px^2 | Dist: {cam_distance} cm")

        annotated_frames.append(ann_frame)

    # Lắp ráp giao diện dựa vào view_mode
    if view_mode == -1:
        top_row = np.hstack((annotated_frames[0], annotated_frames[1]))
        bottom_row = np.hstack((annotated_frames[2], annotated_frames[3]))
        grid_frame = np.vstack((top_row, bottom_row))
        display_frame = cv2.resize(grid_frame, (1280, 720))
    else:
        single_frame = annotated_frames[view_mode]
        display_frame = cv2.resize(single_frame, (1280, 720))

    # Vẽ nút điều khiển
    draw_button(display_frame, "GRID", (20, 20), (120, 60), bg_color=(0, 100, 0))
    if view_mode != -1:
        draw_button(display_frame, "<", (20, 310), (80, 410))
        draw_button(display_frame, ">", (1200, 310), (1260, 410))
        cv2.putText(display_frame, f"ZOOMED: CAMERA {view_mode + 1}", (150, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()