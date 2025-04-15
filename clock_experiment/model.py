import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from data import text_to_indices


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

class SimpleVLMSeq2Seq(nn.Module):
    def __init__(self, input_vocab_size, embed_dim, text_hidden_dim, output_vocab_size, output_seq_len=5):
        """
        A small vision-language model that fuses image and text prompt features
        and then decodes a fixed-length output sequence representing the time (hh:mm).
        
        Args:
          input_vocab_size: Size of the input (prompt) vocabulary.
          embed_dim: Dimensionality of input word embeddings.
          text_hidden_dim: Hidden dimension for the prompt LSTM encoder.
          output_vocab_size: Size of the output (target) vocabulary (e.g., 11 for digits+colon).
          output_seq_len: Fixed length of the output sequence (5 for "hh:mm").
        """
        super(SimpleVLMSeq2Seq, self).__init__()
        # Image branch: a small CNN.
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # [B, 64, 1, 1]
        )
        self.img_fc = nn.Linear(64, 128)  # Map image features to 128 dims.
        
        # Text branch: embedding + LSTM to encode the prompt.
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, text_hidden_dim, batch_first=True)
        # After the LSTM, we take the final hidden state.
        
        # Fusion of image and prompt features.
        # Fused vector dimension = 128 + text_hidden_dim.
        fused_dim = 128 + text_hidden_dim
        
        # Decoder head: outputs logits for each token in the output sequence.
        # We use a single linear layer that maps the fused vector to:
        # (output_seq_len * output_vocab_size) logits.
        self.decoder = nn.Linear(fused_dim, output_seq_len * output_vocab_size)
        self.output_seq_len = output_seq_len
        self.output_vocab_size = output_vocab_size
        
    def forward(self, img, text):
        # Process the image.
        x_img = self.img_encoder(img)           # Shape: (B, 64, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)     # Shape: (B, 64)
        x_img = self.img_fc(x_img)                # Shape: (B, 128)
        
        # Process the text prompt.
        x_text = self.embedding(text)             # Shape: (B, L, embed_dim)
        _, (h_n, _) = self.lstm(x_text)           # h_n: (num_layers, B, text_hidden_dim)
        x_text = h_n[-1]                         # Shape: (B, text_hidden_dim)
        
        # Fuse features.
        fused = torch.cat((x_img, x_text), dim=1)  # Shape: (B, 128 + text_hidden_dim)
        
        # Decode the output sequence in one shot.
        out = self.decoder(fused)  # Shape: (B, output_seq_len * output_vocab_size)
        out = out.view(-1, self.output_seq_len, self.output_vocab_size)
        # Out shape: (B, seq_len, vocab_size)
        return out

############################################
# 5. Training and Evaluation Functions
############################################

def train_model(model, train_loader, criterion, optimizer, device, output_vocab_size):
    model.train()
    running_loss = 0.0
    for images, texts, target_seqs in train_loader:
        images = images.to(device)
        texts = texts.to(device)
        target_seqs = target_seqs.to(device)  # Shape: (B, seq_len)
        
        optimizer.zero_grad()
        outputs = model(images, texts)  # Shape: (B, seq_len, vocab_size)
        # Flatten outputs and targets for cross-entropy:
        loss = criterion(outputs.view(-1, output_vocab_size), target_seqs.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, test_loader, device, output_vocab, output_seq_len):
    model.eval()
    correct_tokens = 0
    total_tokens = 0
    outputs_list = []
    
    # Inference: convert predictions to strings.
    inv_out_vocab = {v: k for k, v in output_vocab.items()}
    
    with torch.no_grad():
        for images, texts, target_seqs in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            target_seqs = target_seqs.to(device)  # Shape: (B, seq_len)
            outputs = model(images, texts)         # Shape: (B, seq_len, vocab_size)
            # Get predicted token indices per time step.
            preds = outputs.argmax(dim=2)          # Shape: (B, seq_len)
            
            # Compute token-level accuracy.
            correct_tokens += (preds == target_seqs).sum().item()
            total_tokens += target_seqs.numel()
            
            # Convert each predicted sequence to a string.
            for seq in preds.cpu().numpy():
                pred_str = "".join([inv_out_vocab[token] for token in seq])
                outputs_list.append(pred_str)
    
    token_acc = correct_tokens / total_tokens
    return token_acc, outputs_list
