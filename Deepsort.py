from deep_sort_realtime.deepsort_tracker import DeepSort

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