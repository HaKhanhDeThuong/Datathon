import cv2
import time
import sys
sys.path.append(r'D:\code_folder\data-code\Datathon2023\git\DatathonFinal\models')
from detector.Yolo_Detector import ObjectDetector

if __name__ == "__main__":
    detector = ObjectDetector()
    cap = cv2.VideoCapture(r"D:\code_folder\data-code\Datathon2023\git\Datathon\data\customer_behaviors_cctv_public_data\front\OneStopMoveEnter2\OneStopMoveEnter2.mpg")

    start_time = time.time()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.image_with_bbox(frame, detector.detect(frame))
        frame_count += 1

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    print(f"FPS: {fps}")

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
