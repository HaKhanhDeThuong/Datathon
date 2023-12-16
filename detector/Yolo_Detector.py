import torch
from torchvision.transforms import functional as F
import cv2
import threading
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from detector.Classifier import Classifier
import supervision as sv
from detector.polygon_generator import Polygon

class Deep_Sort:
    def __init__(self):
       self.tracker = DeepSort(max_age=20)

    def __del__(self):
        pass

    def update_tracker(self, detections, image):
        bbs = []
        for detection in detections:
            box = detection[0].astype(int)
            confidence = detection[2]
            classes = 0
            bb = (box, confidence, classes)
            bbs.append(bb)
        tracks = self.tracker.update_tracks(bbs, frame=image)
        return tracks

class ObjectDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes = [0]
        self.Tracker = Deep_Sort()
        self.classifier = Classifier()
        self.Polygon = Polygon()
        self.zone = sv.PolygonZone(polygon=self.Polygon.cor, frame_resolution_wh=(384, 288))
        self.camera_height = 3.8
        self.focal_length = 1000
        self.child_height = 0.8
        self.adult_height = 1.4


    def estimate_height(self, bbox_height_pixels):
        return (self.camera_height * bbox_height_pixels) / self.focal_length


    def addingID(self, image, detections):
        tracks = self.Tracker.update_tracker(detections, image)
        for track in tracks:
            ltwh = track.to_ltwh()
            track_id = track.track_id
            left, top, width, height = ltwh
            if str(track_id) == '3':
                print(track_id, height)
            cv2.putText(image, "ID: " + str(track_id), (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def addingClass(self, image, detection):
        cls = self.classifier.classify(image, detection)
        if cls == 1:
            context = 'walker'
        else:
            context = 'browser'
        age = 'adult'
        bbox_height_pixels = detection[3] - detection[1]
        estimate_height = self.estimate_height(bbox_height_pixels)
        if estimate_height / 10 > self.child_height and estimate_height / 10 < self.adult_height:
            age = 'child'
        elif estimate_height / 10 > self.adult_height:
            age = 'adult'
        cv2.putText(image, context, (detection[2] + 2, detection[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, age, (detection[2] + 2, detection[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #export json
        return image

    def detect(self, image):
        image = sv.draw_polygon(scene=image, polygon=self.Polygon.cor, color=sv.Color.green(), thickness=1)
        results = self.model(image, size=640)
        detections = sv.Detections.from_yolov5(results)
        mask = self.zone.trigger(detections=detections)
        detections = detections[mask]
        return detections

    def image_with_bbox(self, image, detections):
        for detection in detections:
            box = detection[0].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (244, 208, 105), 1)
            image = self.addingClass(image, box)
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