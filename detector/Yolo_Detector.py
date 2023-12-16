import torch
from torchvision.transforms import functional as F
import cv2
import threading
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from detector.Classifier import Classifier
import supervision as sv
from detector.area_generate import Polygon, PassLine
from scipy.spatial import cKDTree

class Deep_Sort:
    def __init__(self):
       self.tracker = DeepSort(max_age=10)

    def __del__(self):
        pass

    def update_tracker(self, detections, image):
        bbs = []

        for detection in detections:
            box = detection[0].astype(int)
            confidence = detection[2]
            classes = 0
            x, y, right, bottom = box
            width = right - x
            height = bottom - y
            box = [x, y, width, height]
            bb = (box, confidence, classes)
            bbs.append(bb)

        tracks = self.tracker.update_tracks(bbs, frame=image)
        return tracks


class ObjectDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.model.classes = [0]
        self.Tracker = Deep_Sort()
        self.classifier = Classifier()
        self.Polygon = Polygon()
        self.zone = sv.PolygonZone(polygon=self.Polygon.cor, frame_resolution_wh=(384, 288))
        self.camera_height = 3.8
        self.focal_length = 1000
        self.child_height = 0.8
        self.adult_height = 1.4
        self.counting_pass_walker = 0
        self.passLine = PassLine()
        self.counted_ids = []
        self.status = 'cor'
        self.track_history = [0] * 200

    def estimate_height(self, bbox_height_pixels):
        return (self.camera_height * bbox_height_pixels) / self.focal_length


    def addingID(self, image, detections, bbox_coordinates, bbox0, full ):
        w, h = image.shape[:2]
        if bbox0 and bbox_coordinates:
            tracks = self.Tracker.update_tracker(detections, image)
            YourTreeName = cKDTree(bbox_coordinates, leafsize=100)
            Tree0 = cKDTree(bbox0, leafsize=100)
            for index1, track1 in enumerate(tracks):         
                track_id1 = track1.track_id 
                track_id1 = int(track_id1) -1
                if index1 < len(bbox_coordinates):
                    bb1, bb2, bb3, bb4 = full[index1]
                    
                if self.track_history[track_id1] > 4:
                    cv2.putText(image, "group", (int(bb1/w), int(bb2/h - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                x1, y1, x2, y2 = track1.to_ltwh()
                if index1 < len(bbox_coordinates):
                    bb1, bb2, bb3, bb4 = full[index1]
                query_point = np.array([[bb1,  bb2]])
                query0 = np.array([[0, bb2]])
                distances, indices = YourTreeName.query(query_point, k=len(tracks), distance_upper_bound=abs(bb3 - bb1))  
                valid_distances = distances[(distances <= abs(bb3 - bb1)) & (distances != 0) & np.isfinite(distances)]
                distances2, indices2 = Tree0.query(query0, k=len(tracks), distance_upper_bound=(abs(bb4-bb2)/2))
                valid_distances2 = distances2[(distances2 <= abs(bb4-bb2)/2) & (distances2 != 0) & np.isfinite(distances2)]

                
                if np.sum(valid_distances) >= 1:
                    if np.sum(valid_distances2) >= 1:     
                        self.track_history[track_id1] += 1
                else:
                    if(self.track_history[track_id1] > 0):
                        self.track_history[track_id1] -= 1
                if self.track_history[track_id1] > 4:
                    cv2.putText(image, "group", (int(bb1/w), int(bb2/h - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        tracks = self.Tracker.update_tracker(detections, image)
        for track in tracks:
            ltwh = track.to_ltwh()
            track_id = track.track_id

            left, top, width, height = ltwh.astype(int)       
            center_x = int(left + width / 2)
            center_y = int(top + height)
            center_point = (center_x, center_y)

            if self.passLine.is_point_on_line(center_point, status='cor'):
                if track.track_id not in self.counted_ids:
                    self.counting_pass_walker += 1
                    self.counted_ids.append(track.track_id)
                    
            cv2.putText(image, "ID: " + str(track_id), (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def addingClass(self, image, detection):

        cls = self.classifier.classify(image, detection)
        context = 'walker'
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
        image = cv2.line(image, self.passLine.cor_startPoint, self.passLine.cor_endPoint, (255, 0, 0), thickness = 4 )
        image = sv.draw_polygon(scene=image, polygon=self.Polygon.cor, color=sv.Color.green(), thickness=1)
        results = self.model(image, size = 640)
        detections = sv.Detections.from_yolov5(results)
        mask = self.zone.trigger(detections=detections)
        detections = detections[mask]
        return detections

    def image_with_bbox(self, image, detections):
        w, h = image.shape[:2]
        bbox_coordinates = []
        bbox0 = []
        full = []
        for detection in detections:
            box = detection[0].astype(int)
            x = box[0] * w
            y = box[1] * h
            z = box[2] * w
            t = box[3] * h
            bbox_coordinates.append([x, y])
            full.append([x, y, z, t])
            bbox0.append([0,y])          
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (244, 208, 105), 1)
            image = self.addingClass(image, box)
        image = self.addingID(image, detections, bbox_coordinates, bbox0, full)
        cv2.putText(image, f'Count: {self.counting_pass_walker}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        
    def frame_loader(self, path_dir):
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