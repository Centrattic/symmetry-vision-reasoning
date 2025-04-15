from data import generate_dataset, ClockDataset

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from model import SimpleClockCNN, train_model, evaluate_model


# Configuration
n = 5                  # number of discrete positions
image_size = 128       # size of the generated clock image
num_epochs = 50
batch_size = 4         # since the dataset is small
learning_rate = 0.001
num_classes = n * n    # For n=5 -> 25 classes

# Generate dataset images and labels.
images, labels, time_strings = generate_dataset(n=n, size=image_size)
print("Generated dataset of {} images.".format(len(images)))
# Optionally, print the time strings for verification.
for i in range(len(images)):
    print(f"Label {labels[i]}: {time_strings[i]}")

# Define the transform for PyTorch: convert to tensor and normalize.
transform = transforms.Compose([
    transforms.ToTensor(),  # automatically scales pixel values to [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the dataset.
dataset = ClockDataset(images, labels, transform=transform)

# Split into 80% training, 20% test.
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model.
model = SimpleClockCNN(num_classes=num_classes).to(device)

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} -- Loss: {train_loss:.4f} -- Test Acc: {test_acc:.4f}")

print("Training complete.")

save_path = "models/clock_cnn_model.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")


