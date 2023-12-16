import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np

'''
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
'''

class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class Classifier:
    def __init__(self):
        self.model = ResNetModel(num_classes = 2)
        self.model.load_state_dict(torch.load(r'D:\code_folder\data-code\Datathon2023\git\Datathon\classification-model\checkpoint-final.pth'))
        self.transform = transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def cropImage(self, img, bbox):
        x_min, y_min, x_max, y_max = bbox
        left = int(x_min)
        top = int(y_min)
        right = int(x_max)
        bottom = int(y_max)

        cropped_img = img[top:bottom, left:right, :]
        return cropped_img


    def classify(self, image, detection):
        img_ = self.cropImage(image, detection)
        img_ = Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)) # Convert to PIL Image
        img_ = self.transform(img_).unsqueeze(0).to(self.device)
        with torch.no_grad():
            res = self.model(img_).to("cpu")
            cls = torch.argmax(res).item()
        return cls
        