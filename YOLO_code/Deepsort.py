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