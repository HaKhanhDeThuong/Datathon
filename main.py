from detector.Yolo_Detector import ObjectDetector
import threading
import cv2

if __name__ == "__main__":
    Detector = ObjectDetector()
    Detector.mp4_loader(r"D:\code_folder\data-code\Datathon2023\git\Datathon\data-old\customer_behaviors_cctv_mentor_data\cor\OneStopMoveEnter1\OneStopMoveEnter1.mpg")
    #Detector.video_capture()
    #thread = threading.Thread(target=mp4_loader, args=(Detector, "1.mp4"))
    #thread.start()

    #video_thread = threading.Thread(target=video_capture, args=(Detector,))
    #video_thread.start()
    #video_thread.join()