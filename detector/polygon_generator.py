import numpy as np

# initiate polygon zone
class Polygon:
    def __init__(self):
        self.cor = np.array([
            [60,  163],
            [241, 163],
            [322, 265],
            [60, 266]
        ])
        self.front = np.array([
            [64, 145],
            [363, 145],
            [380, 200],
            [20, 201]
        ])
    