import os
from PIL import Image

class CollectObject:
    def __init__(self, dir_):
        self.dir_ =  dir_
        self.cor_dir = os.path.join(self.dir_, 'cor')
        self.front_dir = os.path.join(self.dir_, 'front')
        self.item_list = os.listdir(self.cor_dir)
        self.data_loader_folder = os.path.join(self.dir_, 'data')
        self.data_loader_browser = os.path.join(self.data_loader_folder, 'browser')
        self.data_loader_walker = os.path.join(self.data_loader_folder, 'walker')
        os.makedirs(self.data_loader_folder, exist_ok=True)
        os.makedirs(self.data_loader_browser, exist_ok=True)
        os.makedirs(self.data_loader_walker, exist_ok=True)
        self.cnt_browser = 0
        self.cnt_walker = 0

    def cropImage(self, img, bbox):
        xc, yc, w, h = bbox
        left = int(xc - w / 2)
        top = int(yc - h / 2)
        right = int(xc + w / 2)
        bottom = int(yc + h / 2)
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img

    def readFile(self, label_, image_):
        file = open(label_, 'r')
        original_img = Image.open(image_)

        for line in file.readlines():
            line = line.strip().replace(' ', '')
            line = line.split(',')
            bbox = [int(i) for i in line[:4]]
            cls = line[4]
            img_cropped = self.cropImage(original_img, bbox)
            if img_cropped.width * img_cropped.height >= 600: #DPI 
                if cls == 'browser':
                    img_cropped.save(os.path.join(self.data_loader_browser, f"{self.cnt_browser}.jpg"))
                    self.cnt_browser += 1
                elif cls == 'walker':
                    img_cropped.save(os.path.join(self.data_loader_walker, f"{self.cnt_walker}.jpg"))
                    self.cnt_walker += 1

    def loadFolder(self):
        for item in self.item_list:
            item_cor_folder = os.path.join(self.cor_dir, item)
            item_front_folder = os.path.join(self.front_dir, item)

            item_cor_label_dir = os.path.join(item_cor_folder, 'label')
            item_cor_image_dir = os.path.join(item_cor_folder, 'frame')
            cor_labels_list = os.listdir(item_cor_label_dir)
            cor_images_list = os.listdir(item_cor_image_dir)

            item_front_label_dir = os.path.join(item_front_folder, 'label')
            item_front_image_dir = os.path.join(item_front_folder, 'frame')

            for li in os.listdir(item_front_label_dir):
                label_ = os.path.join(item_front_label_dir, li)
                image_ = os.path.join(item_front_image_dir, li).replace('.txt', '.jpg')
                self.readFile(label_, image_)


collect_obj = CollectObject(r'D:\code_folder\data-code\Datathon2023\git\Datathon\data\customer_behaviors_cctv_public_data')
collect_obj.loadFolder()