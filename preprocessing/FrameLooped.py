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
    images_list = os.listdir(data_dir)
    img_pre = cv2.imread(os.path.join(data_dir, images_list[0])) 
    for dir_ in images_list[1:]: ##check previous and current image (range(1))
        file = os.path.join(data_dir, dir_)
        img_cur = cv2.imread(file)
        if is_looped(img_pre, img_cur):
            print(f'{dir_} is looped. Deleting the current file.')
            os.remove(file)
            label_file = file.replace('frame', 'label')
            label_file = label_file.replace('.jpg', '.txt')
            os.remove(label_file)
        else:
            img_pre = img_cur

if __name__ == "__main__":
    dir_ =  r'D:\code_folder\data-code\Datathon2023\git\Datathon\data'
    dir_ = os.path.join(dir_, os.listdir(dir_)[0])
    cor_dir = os.path.join(dir_, 'cor')
    front_dir = os.path.join(dir_, 'front')
    item_list = os.listdir(cor_dir)

    for item in item_list:
        item_cor_folder = os.path.join(cor_dir, item)
        item_front_folder = os.path.join(front_dir, item)
        item_cor_image_dir = os.path.join(item_cor_folder, 'frame')
        item_front_image_dir = os.path.join(item_front_folder, 'frame')
        process(item_cor_image_dir)
        process(item_front_image_dir)