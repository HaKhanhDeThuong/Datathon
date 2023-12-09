from Object_Detector import ObjectDetector, video_capture, video_show
import threading
import cv2

if __name__ == "__main__":
    Detector = ObjectDetector()
    video_show(Detector, r"D:\code_folder\data-code\Datathon2023\dataset\data\customer_behaviors_cctv_snapshot_data\cor\OneStopNoEnter1")

    #video_thread = threading.Thread(target=video_capture, args=(Detector,))
    #video_thread.start()
    #video_thread.join()