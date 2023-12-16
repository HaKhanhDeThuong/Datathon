import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def train_model(model, train_loader, num_epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print batch progress
            if batch_idx % 10 == 9:  # Print every 10 batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10}")
                running_loss = 0.0

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

def __init__ == "__main__":
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root='D:\code_folder\data-code\Datathon2023\git\Datathon\data\customer_behaviors_cctv_public_data\data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_classes = 2

    for inputs, labels in train_loader:
        print(f"Batch size: {inputs.size(0)}")
        print("Labels:", labels)
        for i in range(min(4, inputs.size(0))): 
            image = inputs[i].permute(1, 2, 0).numpy()  
            label = labels[i].item()
            plt.subplot(1, 4, i + 1)
            plt.imshow(image)
            plt.title(f"Class: {label}")
            plt.axis("off")
        plt.show()
        break  

    model = CustomResNet50(num_classes = 2)
    train_model(model, train_loader, num_epochs=5, learning_rate=0.001)
    torch.save(model.state_dict(), 'checkpoint-final.pth')