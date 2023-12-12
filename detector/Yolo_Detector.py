import torch
from torchvision.transforms import functional as F
import cv2
import threading
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

class Deep_Sort:
    def __init__(self):
       self.tracker = DeepSort(max_age=5)

    def __del__(self):
        pass

    def update_tracker(self, detections, image):
        bbs = []
        for detection in detections.xyxy[0]:
            box = detection[:4].cpu().numpy().astype(int)
            confidence = detection[5].cpu().numpy().astype(float)
            classes = 0
            bb = (box, confidence, classes)
            bbs.append(bb)
        tracks = self.tracker.update_tracks(bbs, frame=image)
        return tracks

class ObjectDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        self.model.classes = [0]
        self.Tracker = Deep_Sort()

    def addingID(self, image, detections):
        tracks = self.Tracker.update_tracker(detections, image)
        for track in tracks:
            ltwh = track.to_ltwh()
            track_id = track.track_id
            left, top, width, height = ltwh
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
            
    def mp4_loader(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.image_with_bbox(frame, self.detect(frame))
            cv2.imshow('mp4Loader', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
    def frame_loader(self, frames_path):
        list_item = os.listdir(path_dir) 
        for item in list_item:
            image_path = os.path.join(path_dir, item)
            frame = cv2.imread(image_path)    
            frame = self.image_with_bbox(frame, self.detect(frame))
            cv2.imshow('FrameShow', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        
    def video_capture(self):
        cap = cv2.VideoCapture(0) 

        while True:
            ret, frame = cap.read()
            frame = self.image_with_bbox(frame, self.detect(frame))
            cv2.imshow('Camera', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()