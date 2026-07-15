import cv2
import numpy as np
import serial
import threading
from ultralytics import YOLO

# ==========================================
# 1. ĐỌC VÀ TÁCH CHUỖI SERIAL (x y z t)
# ==========================================
# Mảng lưu khoảng cách [x, y, z, t] tương ứng với CAM 1, 2, 3, 4
current_distances = [-1.0, -1.0, -1.0, -1.0] 
sensor_labels = ['x', 'y', 'z', 't'] # Tên các cảm biến tương ứng

def read_serial_data(port='/dev/arduino_nano', baudrate=9600):
    global current_distances
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[HỆ THỐNG] Đã kết nối thành công với Arduino tại {port}")
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                parts = line.split()
                # Đảm bảo nhận đủ 4 giá trị x y z t
                if len(parts) == 4:
                    try:
                        # Tách chuỗi thành 4 biến float độc lập và lưu vào mảng toàn cục
                        current_distances = [float(p) for p in parts]
                    except ValueError:
                        pass # Bỏ qua nếu dòng dữ liệu bị nhiễu ký tự
    except Exception as e:
        print(f"[LỖI] Không thể kết nối cổng Serial: {e}")

# Khởi chạy luồng đọc Serial song song
serial_thread = threading.Thread(target=read_serial_data, daemon=True)
serial_thread.start()

# ==========================================
# 2. KHỞI TẠO CAMERA VÀ MODEL OBB
# ==========================================
model = YOLO('yolo11n-obb.pt', task='obb') 

cap = cv2.VideoCapture("/dev/cam_xvr", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("[LỖI] Không thể mở luồng USB Capture!")
    exit()

# ==========================================
# 3. GIAI ĐOẠN CHỜ KHUNG THÔNG BÁO BIẾN MẤT
# ==========================================
print("[HỆ THỐNG] Đang giám sát khung thông báo từ đầu ghi...")
clear_frames_count = 0 

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Lấy mẫu một ô vuông 50x50 pixel ở chính giữa màn hình (nơi hiện Popup)
    roi = frame[mid_h - 25 : mid_h + 25, mid_w - 25 : mid_w + 25]
    
    # Chuyển vùng lấy mẫu sang ảnh xám và tính độ sáng trung bình
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_roi)

    # Nếu khung thông báo màu trắng biến mất, độ sáng trung bình vùng này sẽ giảm sâu (< 180)
    if avg_brightness < 180:
        clear_frames_count += 1
    else:
        clear_frames_count = 0 # Reset nếu khung trắng xuất hiện trở lại

    # Vẽ ô vuông đỏ giám sát lên màn hình chờ để bạn dễ quan sát vị trí lấy mẫu
    cv2.rectangle(frame, (mid_w - 25, mid_h - 25), (mid_w + 25, mid_h + 25), (0, 0, 255), 2)
    cv2.putText(frame, f"Cho XVR on dinh... Do sang ROI: {avg_brightness:.0f}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Startup Waiting", frame)

    # Nếu sạch bóng khung trắng trong 15 khung hình liên tiếp, tự động chuyển trạng thái
    if clear_frames_count > 15:
        print("[HỆ THỐNG] Đã xác nhận khung thông báo biến mất. Bắt đầu chia 4 luồng AI!")
        cv2.destroyWindow("Startup Waiting")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ==========================================
# 4. GIAI ĐOẠN TÁCH 4 LUỒNG VÀ GÁN KHOẢNG CÁCH CHO TỪNG OBB LỚN NHẤT
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2

    # Cắt ảnh Full HD thành 4 phần tư độc lập
    cam1 = frame[0:mid_h, 0:mid_w]           # Góc trên trái -> Gắn với cảm biến x
    cam2 = frame[0:mid_h, mid_w:w]           # Góc trên phải -> Gắn với cảm biến y
    cam3 = frame[mid_h:h, 0:mid_w]           # Góc dưới trái -> Gắn với cảm biến z
    cam4 = frame[mid_h:h, mid_w:w]           # Góc dưới phải -> Gắn với cảm biến t

    # Gom 4 khung hình chạy AI song song (Batch Inference) để tối ưu phần cứng Jetson
    batch_frames = [cam1, cam2, cam3, cam4]
    results = list(model(batch_frames, stream=True, conf=0.5, verbose=False))

    # Xử lý kết quả độc lập cho từng góc camera
    for i, result in enumerate(results):
        # Lấy hình ảnh đã được vẽ sẵn khung OBB mặc định từ YOLO
        annotated_frame = result.plot()
        
        # Đánh dấu tên Camera cứng lên góc trái màn hình
        cv2.putText(annotated_frame, f"CAM {i+1}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Trích xuất giá trị khoảng cách tương ứng của camera này
        cam_distance = current_distances[i]
        sensor_name = sensor_labels[i]

        max_area = 0
        closest_obb_corners = None
        closest_class_id = -1

        # Tìm kiếm Oriented Bounding Box (OBB) lớn nhất CHỈ TRONG camera này
        if result.obb is not None and len(result.obb) > 0:
            for obb in result.obb:
                # Trích xuất (x, y, width, height, rotation)
                _, _, bw, bh, _ = obb.xywhr[0].cpu().numpy()
                area = bw * bh 
                
                if area > max_area:
                    max_area = area
                    closest_class_id = int(obb.cls[0].item())
                    # Lấy tọa độ 4 góc xoay của hình chữ nhật OBB (Ma trận 4x2)
                    closest_obb_corners = obb.xyxyxyxy[0].cpu().numpy().astype(int)

            # Gán dữ liệu khoảng cách lên OBB lớn nhất vừa tìm được
            if closest_obb_corners is not None and cam_distance > 0:
                # Thuật toán tìm điểm cao nhất (Y nhỏ nhất) trong 4 góc OBB để đặt text không bị đè lên hình
                top_point = closest_obb_corners[np.argmin(closest_obb_corners[:, 1])]
                x_text, y_text = top_point[0], top_point[1] - 10
                
                # Vẽ text khoảng cách trực tiếp lên khung OBB
                text = f"DIST: {cam_distance:.1f} cm"
                cv2.putText(annotated_frame, text, (x_text, y_text), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Trích xuất tên nhãn đối tượng (ví dụ: 'person')
                class_name = model.names[closest_class_id]
                
                # Log chi tiết trạng thái hoạt động độc lập ra Console
                print(f"[LOG] CAM {i+1} | {class_name.upper()} gan nhat | S-Area: {max_area:.0f} px^2 | Khoang cach ({sensor_name}): {cam_distance:.1f} cm")
        else:
            # Nếu không phát hiện đối tượng nhưng cảm biến siêu âm báo khoảng cách nguy hiểm
            if cam_distance > 0 and cam_distance < 150: # Ngưỡng cảnh báo an toàn 1.5m
                print(f"[LOG] CAM {i+1} | Khong thay vat the | Canh bao vat can vo hinh ({sensor_name}): {cam_distance:.1f} cm")

        # Hiển thị đồng thời 4 luồng camera riêng biệt
        cv2.imshow(f"Luong CAM {i+1}", annotated_frame)

    # Thoát chương trình khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
