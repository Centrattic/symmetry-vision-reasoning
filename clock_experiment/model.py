import torch
import torch.nn as nn


class SimpleClockCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleClockCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2)   # [B, 64, 16, 16]
        )
        # Flatten: 64 x 16 x 16 = 16384 features.
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

###############################
# 4. Training Script
###############################

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy
