import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_bbox_heatmap(image, boxes, alpha=0.6):
    """
    Vẽ bounding boxes thành heatmap lên ảnh.

    Parameters:
        - image: Ảnh nguồn (numpy array).
        - boxes: Danh sách các bounding boxes, mỗi box là một tuple (x_min, y_min, x_max, y_max).
        - alpha: Độ trong suốt của heatmap.

    Returns:
        - heatmap: Ảnh heatmap với bounding boxes.
    """
    heatmap = np.zeros_like(image, dtype=np.float32)

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        heatmap[y_min:y_max, x_min:x_max, :] += 1

    heatmap = np.clip(heatmap, 0, 1)
    heatmap = np.uint8(255 * alpha * heatmap / np.max(heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return result

# Example usage:
image_path = r"D:\code_folder\data-code\Datathon2023\code\data\customer_behaviors_cctv_mentor_data\front\OneLeaveShop1\frame\OneLeaveShop1front0000.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Example bounding boxes (replace with your own):
boxes = [(376, 279, 14, 10), (102, 27, 14, 38)]

result_image = draw_bbox_heatmap(image, boxes)

# Display the result
plt.imshow(result_image)
plt.axis('off')
plt.show()
