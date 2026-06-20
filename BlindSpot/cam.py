import cv2
from ultralytics import YOLO

# 1. Tải mô hình YOLOv11 Segmentation (Sẽ tự động tải file .pt về nếu chưa có)
# Bạn có thể dùng model lớn hơn trên laptop như 'yolo11s-seg.pt' hoặc 'yolo11m-seg.pt'
model = YOLO(r'E:\Downloads\Data road-20260508T202317Z-3-001\weight\best.pt')

# 2. Mở kết nối camera
# LƯU Ý TRÊN LAPTOP:
# Số 0 thường là webcam tích hợp của laptop.
# USB Video Capture của bạn lúc này thường sẽ mang ID là 1 hoặc 2.
camera_id = 1
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

# Ép độ phân giải về 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window_name = "YOLOv11 Segmentation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Cho phép người dùng kéo thả viền cửa sổ
cv2.resizeWindow(window_name, 1280, 720)

if not cap.isOpened():
    print(f"Không thể mở camera số {camera_id}.")
    exit()

print("Bắt đầu luồng Segmentation. Nhấn 'q' để thoát.")


while True:
    success, frame = cap.read()
    if not success:
        print("Lỗi đọc khung hình.")
        break

    # 3. Chạy suy luận (Segmentation)
    # Tùy chỉnh retina_masks=True nếu bạn muốn mask vẽ ra sắc nét và chính xác đến từng pixel ở độ phân giải cao
    results = model(frame, stream=True, conf=0.5, retina_masks=True, verbose=False)

    # 4. Hậu xử lý & Hiển thị
    for result in results:
        # Hàm plot() sẽ tự động phủ màu (mask) và vẽ viền (polygon) lên các vật thể nhận diện được
        annotated_frame = result.plot()

        # (Nâng cao) Nếu bạn muốn trích xuất tọa độ để làm logic bên ngoài:
        # if result.masks is not None:
        #     # result.masks.xy trả về danh sách tọa độ các đa giác (polygons)
        #     for polygon in result.masks.xy:
        #         pass # Viết code kiểm tra diện tích hoặc vị trí tại đây

    # Hiển thị
    cv2.imshow("YOLOv11 Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
