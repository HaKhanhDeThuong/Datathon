import numpy as np

# initiate polygon zone
class Polygon:
    def __init__(self):
        self.cor = np.array([
            [60,  148],#
            [241, 146],
            [322, 265],
            [60, 266]#
        ])
        self.front = np.array([
            [64, 145 - 15],#
            [363, 145 - 15],#
            [380, 200],
            [20, 201]
        ])
    
class PassLine:
    def __init__(self):
        self.cor_startPoint = (60, 148)
        self.cor_endPoint = (60, 266)
        self.front_startPoint = (214, 145 - 15)
        self.front_endPoint = (327, 145 - 15)
    def is_point_on_line(self, point, status):
        x, y = point

        if status == 'cor':
            x1, y1 = self.cor_startPoint
            x2, y2 = self.cor_endPoint
        else:
            x1, y1 = self.front_startPoint
            x2, y2 = self.front_endPoint

        if x2 - x1 == 0:  # Vertical line
            return abs(x - x1) < 5  # Check if the x-coordinate is close to the line
        else:
            slope = (y2 - y1) / (x2 - x1)
            expected_y = y1 + slope * (x - x1)
            tolerance = 10
            return abs(y - expected_y) < tolerance