import torch
from torchvision.transforms import functional as F
import cv2
import threading
from Deepsort import Deep_Sort
import numpy as np
import os

class ObjectDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes = [0]
        #self.model.eval()
        self.Tracker = Deep_Sort()

    def addingID(self, image, detections):
        tracks = self.Tracker.update_tracker(detections, image)
        for track in tracks:
            ltwh = track.to_ltwh()
            track_id = track.track_id
            left, top, width, height = ltwh
            # ID ASSIGNMENT
            cv2.putText(image, "ID: " + str(track_id), (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def detect (self, image):
        detections = self.model(image)
        return detections

    def image_with_bbox(self, image, detections):
        for detection in detections.xyxy[0]:
            box = detection[:4].cpu().numpy().astype(int)
            confidence = detection[4].cpu().numpy().astype(float)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        image = self.addingID(image, detections)
        return image

    def show_image(self, image):
        cv2.imshow('showImage', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def save_detected_video(detector, video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video parameters (frame size, frame rate)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create VideoWriter to write the output video
    output_path = os.path.join(output_folder, 'output.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        frame_with_bbox = detector.image_with_bbox(frame, detections)

        # Write the frame with detections to the output video
        out.write(frame_with_bbox)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
detector = ObjectDetector()
# Choose either video_capture or save_detected_video based on your needs
# video_capture(detector)
# save_detected_video(detector, 'path_to_video.mp4', 'output_folder')
