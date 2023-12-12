from Object_Detector import ObjectDetector, video_capture, video_show, mp4_loader
import threading
import cv2

if __name__ == "__main__":
    Detector = ObjectDetector()
    mp4_loader(Detector, r"1.mp4")

    #video_thread = threading.Thread(target=video_capture, args=(Detector,))
    #video_thread.start()
    #video_thread.join()