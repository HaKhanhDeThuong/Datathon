import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import threading
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

#---------------------------------------------------------------------#


class Deep_Sort:
    def __init__(self):
       self.tracker = DeepSort(max_age=5)

    def __del__(self):
        pass

    def update_tracker(self, detections, image):
        bbs = []
        for d in detections: 
            box = d[0]
            conf = d[1]
            classes = d[2]
            bb = (box, conf, classes)
            bbs.append(bb)
        tracks = self.tracker.update_tracks(bbs, frame=image)
        return tracks
        
class ObjectDetector:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.Tracker = Deep_Sort()

    def preprocess_image(self, image):
        image_resized = cv2.resize(image, (128, 128))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(self.device)
        return image_tensor

    def addingID(self, image, detection):
        tracks = self.Tracker.update_tracker(detection, image)
        for track in tracks:
            ltwh = track.to_ltwh()
            track_id = track.track_id
            left, top, width, height = ltwh
            # ID ASSIGMENT
            cv2.putText(image, "ID: " + str(track_id), (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

    def image_with_bbox(self, image, labels, scores, boxes):
        detection = []
        for label in labels:
            score = scores[label[0]]
            if float(score) > 0.7: #optional
                box = [int(coord) for coord in boxes[label[0]]]
                detection.append([box, float(score), 1])           
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        image = self.addingID(image, detection)
        return image

    def detect_objects(self, image):
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(image_tensor)

        labels = [(index, label) for index, label in enumerate(predictions[0]['labels']) if label == 1]
        scores = predictions[0]['scores']
        return [labels, scores, predictions[0]['boxes']]

    def show_image(self, image):
        cv2.imshow('showImage', image)
        if 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def video_capture(self):
        cap = cv2.VideoCapture(0) 

        while True:
            ret, frame = cap.read()
            result = self.detect_objects(frame)
            frame = self.image_with_bbox(frame, result[0], result[1], result[2])
            cv2.imshow('Camera', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def video_show(self, path_dir): #path of folder contain frames
        list_item = os.listdir(path_dir) 
        for item in list_item:
            image_path = os.path.join(path_dir, item)
            frame = cv2.imread(image_path)    
            result = self.detect_objects(frame)
            frame = self.image_with_bbox(frame, result[0], result[1], result[2])
            cv2.imshow('FrameShow', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
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
            print('1')
            result = self.detect_objects(frame)
            print('2')
            frame = self.image_with_bbox(frame, result[0], result[1], result[2])
            print('3')
            cv2.imshow('mp4Loader', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

