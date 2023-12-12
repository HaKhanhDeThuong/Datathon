import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def is_looped(img_pre, img_cur):
    img_pre = img_pre[0:9, 0:209]
    img_cur = img_cur[0:9, 0:209]

    similarity_ratio = np.sum(img_pre == img_cur) / np.prod(img_pre.shape)

    return similarity_ratio == 1

def process(data_dir):
    img_pre = cv2.imread(os.path.join(data_dir, images_list[199])) 
    for dir_ in images_list: ##check previous and current image (range(1))
        file = os.path.join(data_dir, dir_)
        img_cur = cv2.imread(file)
        if is_looped(img_pre, img_cur):
            print(f'{dir_} is looped')
        img_pre = img_cur

if __name__ == "__main__":
    data_dir = r'data/cor/OneStopNoEnter1'
    images_list = os.listdir(data_dir)
    process(data_dir)