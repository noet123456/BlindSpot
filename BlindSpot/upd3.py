import cv2
import numpy as np
import serial
import threading
import os
import time
from ultralytics import YOLO

# ==========================================
# 1. ĐỌC TÍN HIỆU SERIAL (X, Y, Z, T TỪ 1 CỔNG)
# ==========================================
dist_x = dist_y = dist_z = dist_t = -1.0

def read_serial_data(port='/dev/ttyUSB0', baudrate=9600):
    global dist_x, dist_y, dist_z, dist_t
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[HỆ THỐNG] Đã kết nối Cổng ({port}) -> Đọc X, Y, Z, T")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data = line.split() 
                # Kiểm tra xem có đúng 4 giá trị được gửi lên không
                if len(data) == 4:
                    try:
                        dist_x = float(data[0])
                        dist_y = float(data[1])
                        dist_z = float(data[2])
                        dist_t = float(data[3])
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[LỖI] Không thể kết nối Cổng ({port}): {e}")

# Khởi chạy 1 luồng đọc Serial duy nhất
serial_thread = threading.Thread(target=read_serial_data, args=('/dev/ttyUSB0', 9600), daemon=True)
serial_thread.start()

# ==========================================
# CẤU HÌNH CẢNH BÁO ÂM THANH
# ==========================================
wlv1 = 150.0
wlv2 = 100.0 
last_alert_time = 0

def warning(delay):
    global last_alert_time
    # Chỉ phát âm thanh 2 giây một lần để tránh đè tiếng
    if time.time() - last_alert_time > 1.0:
        # Thay 'canh_bao.wav' bằng đường dẫn tới file âm thanh của bạn (VD: '/home/hust/canh_bao.wav')
        os.system('aplay canh_bao.wav > /dev/null 2>&1 &')
        last_alert_time = time.time()

# ==========================================
# 2. KHỞI TẠO CAMERA VÀ MODEL
# ==========================================
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
# 4. THIẾT LẬP GIAO DIỆN (GUI) & FULLSCREEN
# ==========================================
WINDOW_NAME = "Jetson Orin - Smart Surveillance Dashboard"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

view_mode = -1

def mouse_click(event, mouse_x, mouse_y, flags, param):
    global view_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= mouse_x <= 120 and 20 <= mouse_y <= 60:
            view_mode = -1
        if view_mode == -1 and mouse_y > 80:
            mid_x, mid_y = 1280 // 2, 720 // 2
            if mouse_x < mid_x and mouse_y < mid_y: view_mode = 0  
            elif mouse_x >= mid_x and mouse_y < mid_y: view_mode = 1  
            elif mouse_x < mid_x and mouse_y >= mid_y: view_mode = 2  
            elif mouse_x >= mid_x and mouse_y >= mid_y: view_mode = 3  
        if view_mode != -1:
            if 20 <= mouse_x <= 80 and 310 <= mouse_y <= 410:  
                view_mode = (view_mode - 1) % 4
            elif 1200 <= mouse_x <= 1260 and 310 <= mouse_y <= 410:  
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

# ==========================================
# KHỞI TẠO BỘ GHI VIDEO KÉP (Thêm comment hoặc bỏ để tắt bật quay màn hình)
# ==========================================
print("[HỆ THỐNG] Đang khởi tạo bộ ghi video (Giao diện & Gốc)...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_ui_video = cv2.VideoWriter('record_giao_dien.mp4', fourcc, 20.0, (1280, 720))
out_raw_video = cv2.VideoWriter('record_camera_goc.mp4', fourcc, 20.0, (1920, 1080))


# ==========================================
# VÒNG LẶP CHÍNH (AI & ÁNH XẠ CẢM BIẾN)
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        break
    # Thêm comment hoặc bỏ để tắt bật quay màn hình
    out_raw_video.write(frame)

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    cam1 = frame[0:mid_h, 0:mid_w]
    cam2 = frame[0:mid_h, mid_w:w]
    cam3 = frame[mid_h:h, 0:mid_w]
    cam4 = frame[mid_h:h, mid_w:w]

    batch_frames = [cam1, cam2, cam3, cam4]
    
    results = []
    global_total_objects = 0 
    maximum_warning_objects = 20
    for cam in batch_frames:
        res = model(cam, conf=0.5, verbose=False)[0]
        results.append(res)
        global_total_objects += len(res.boxes) if res.boxes is not None else 0

    global_max_area = 0
    target_cam_idx = -1
    target_rect_corners = None
    target_class_id = -1

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
    
    # BẢN ĐỒ ÁNH XẠ: Cam 1 -> x, Cam 2 -> y, Cam 3 -> z, Cam 4 -> t
    sensor_values = [dist_x, dist_y, dist_z, dist_t]

    for i, result in enumerate(results):
        ann_frame = result.plot()
        local_count = len(result.boxes) if result.boxes is not None else 0
        cv2.putText(ann_frame, f"CAM {i + 1} | Obj: {local_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # Lấy khoảng cách tương ứng với luồng Camera hiện tại
        cam_distance = sensor_values[i]

        if i == target_cam_idx and target_rect_corners is not None and cam_distance > 0:
            cv2.drawContours(ann_frame, [target_rect_corners], 0, (0, 255, 0), 2)
            top_point = target_rect_corners[np.argmin(target_rect_corners[:, 1])]
            cv2.putText(ann_frame, f"{cam_distance} cm", (top_point[0], top_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            class_name = model.names[target_class_id]
            print(f"[CẢNH BÁO] {class_name.upper()} tại CAM {i + 1} | Dist: {cam_distance}cm | Total: {global_total_objects}")

            if wlv2 < current_distance <= wlv1 and global_total_objects < maximum_warning_objects:
                warning(1.0)
            elif 0 < current_distance <= wlv2 and global_total_objects < maximum_warning_objects:
                warning(0.5)


        annotated_frames.append(ann_frame)

    if view_mode == -1:
        top_row = np.hstack((annotated_frames[0], annotated_frames[1]))
        bottom_row = np.hstack((annotated_frames[2], annotated_frames[3]))
        grid_frame = np.vstack((top_row, bottom_row))
        display_frame = cv2.resize(grid_frame, (1280, 720)) 
    else:
        single_frame = annotated_frames[view_mode]
        display_frame = cv2.resize(single_frame, (1280, 720))

    draw_button(display_frame, "GRID", (20, 20), (120, 60), bg_color=(0, 100, 0))
    cv2.putText(display_frame, f"TOTAL SYSTEM OBJECTS: {global_total_objects}", (800, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if view_mode != -1:
        draw_button(display_frame, "<", (20, 310), (80, 410))
        draw_button(display_frame, ">", (1200, 310), (1260, 410))
        cv2.putText(display_frame, f"ZOOMED: CAMERA {view_mode + 1}", (150, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # Thêm hoặc bỏ comment để tắt bật quay màn hình
    out_ui_video.write(display_frame)
    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[HỆ THỐNG] Đang đóng các file video và thoát chương trình...")
cap.release()
# Thêm comment để tắt bật quay màn hình
out_ui_video.release() 
out_raw_video.release()
# ======================================================================
cv2.destroyAllWindows()