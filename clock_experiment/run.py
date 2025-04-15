
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
from PIL import Image

from data import generate_dataset, ClockVLMDataset
from model import SimpleVLM, train_model, evaluate_model


# Configuration
n = 5                  # number of discrete positions
image_size = 128       # size of the generated clock image
num_epochs = 100
batch_size = 4         # since the dataset is small
learning_rate = 0.0005
num_classes = n * n    # For n=5 -> 25 classes

# Text prompt used for every sample.
prompt = "Tell me the time on the clock"
# Build a simple vocabulary from the prompt.
# The prompt tokenized (lowercase) is: ["tell", "me", "the", "time", "on", "the", "clock"]
# Our vocab maps unique tokens to indices.
vocab = {"tell": 0, "me": 1, "the": 2, "time": 3, "on": 4, "clock": 5}


# Generate dataset images and labels.
images, labels, time_strings = generate_dataset(n=n, size=image_size)

print("Generated dataset of {} images.".format(len(images)))

save_folder = f"clock{n}_images"
os.makedirs(save_folder, exist_ok=True)

# Assume you already have the lists: images (numpy arrays) and time_strings.
# For each image, create a filename and save the image.
for i, (img_array, time_str) in enumerate(zip(images, time_strings)):
    # Modify the time string to be filename-friendly (replace ':' with '_').
    safe_time_str = time_str.replace(":", "_")
    filename = f"clock_{i}_{safe_time_str}.png"
    filepath = os.path.join(save_folder, filename)
    # Convert the numpy array to a PIL Image and save.
    Image.fromarray(img_array).save(filepath)

print(f"All images have been saved to the folder: {save_folder}")

# # Optionally, print the time strings for verification.
# for i in range(len(images)):
#     print(f"Label {labels[i]}: {time_strings[i]}")

# Define the transform for PyTorch: convert to tensor and normalize.
transform = transforms.Compose([
    transforms.ToTensor(),  # scales to [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the Vision-Language dataset.
dataset = ClockVLMDataset(images, labels, prompt, vocab, transform=transform)

# Split dataset into 80% training and 20% testing.
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters for the text branch.
vocab_size = len(vocab)
embed_dim = 32         # Dimension for word embeddings
text_hidden_dim = 128  # Hidden size for the LSTM; we assume 128 for fusion

# Create the model.
model = SimpleVLM(vocab_size, embed_dim, text_hidden_dim, num_classes).to(device)

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} -- Loss: {train_loss:.4f} -- Test Acc: {test_acc:.4f}")

print("Training complete.")

# -----------------------------------
# Save the trained model's state.
save_path = f"clock_vlm_model_{n}_{image_size}_{num_epochs}_{batch_size}_{learning_rate}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
