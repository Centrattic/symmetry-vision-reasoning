import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset



class ClockVLMDataset(Dataset):
    def __init__(self, images, labels, prompt, vocab, transform=None):
        """
        A PyTorch Dataset for clock images with a fixed text prompt.
        
        Args:
            images (list of np.array): List of generated clock images.
            labels (list of int): Integer label for each image.
            prompt (str): The text prompt (e.g., "Tell me the time on the clock").
            vocab (dict): A dictionary mapping tokens to indices.
            transform: Optional torchvision transforms.
        """
        self.images = images
        self.labels = labels
        self.prompt = prompt
        self.vocab = vocab
        self.transform = transform
        # Precompute text indices from the prompt (same for every sample).
        self.text_indices = torch.tensor(text_to_indices(prompt, vocab), dtype=torch.long)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # Return image, text, and label. The text is the same for each sample.
        return image, self.text_indices, label

###############################
# 4. Vision-Language Model Definition
###############################

class SimpleVLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, text_hidden_dim, num_classes):
        """
        A simple Vision-Language Model that fuses image and text features.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality for word embeddings.
            text_hidden_dim (int): Hidden size for the LSTM text encoder.
            num_classes (int): Number of output classes.
        """
        super(SimpleVLM, self).__init__()
        # Image branch: a small CNN encoder.
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # [B, 16, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # [B, 32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))                              # [B, 64, 1, 1]
        )
        self.img_fc = nn.Linear(64, 128)  # Map image features to 128 dimensions
        
        # Text branch: an embedding layer followed by an LSTM.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, text_hidden_dim, batch_first=True)
        # We assume text_hidden_dim is 128 (or you can adjust fusion accordingly).
        
        # Fusion: Concatenate image and text features and run through a classifier.
        self.classifier = nn.Sequential(
            nn.Linear(128 + text_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, img, text):
        # Process image.
        x_img = self.img_encoder(img)          # shape: (B, 64, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)    # shape: (B, 64)
        x_img = self.img_fc(x_img)               # shape: (B, 128)
        
        # Process text.
        # 'text' is expected to be of shape (B, L) where L is the sequence length.
        x_text = self.embedding(text)            # shape: (B, L, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x_text)   # h_n shape: (num_layers, B, text_hidden_dim)
        x_text = h_n[-1]                         # use the final hidden state, shape: (B, text_hidden_dim)
        
        # Fuse image and text features.
        fused = torch.cat((x_img, x_text), dim=1)  # shape: (B, 128 + text_hidden_dim)
        out = self.classifier(fused)
        return out

###############################
# 5. Training and Evaluation Functions
###############################

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, texts, labels in train_loader:
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, texts)
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
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy
