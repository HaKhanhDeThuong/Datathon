import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


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

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1

def plot_metrics(accuracy, precision, recall, f1):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]

    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Assuming scores are between 0 and 1
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(root=r'D:\code_folder\data-code\Datathon2023\git\Datathon\data\customer_behaviors_cctv_public_data\data\test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    classifier = Classifier()

    accuracy, precision, recall, f1 = evaluate_model(classifier.model, test_loader, classifier.device)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    plot_metrics(accuracy, precision, recall, f1)
