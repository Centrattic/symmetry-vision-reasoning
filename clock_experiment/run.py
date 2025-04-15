
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
from PIL import Image

from data import generate_dataset, ClockVLMDataset, generate_clock_image
from model import SimpleVLMSeq2Seq, train_model, evaluate_model


# Configuration
n = 10                      # Number of discrete positions (creates n^2 images).
image_size = 128            # Resolution of generated images.
num_epochs = 200            # Number of training epochs.
batch_size = 4              # Batch size.
learning_rate = 0.001       # Learning rate.
output_seq_len = 5          # Length of the time string ("hh:mm").

# Define the input prompt.
prompt = "Tell me the time on the clock in hh:mm format"

# Define the input vocabulary for the prompt.
# (This is a minimal vocabulary for tokenizing the prompt.)
input_vocab_tokens = ["tell", "me", "the", "time", "on", "clock", "in", "hh:mm", "format"]
input_vocab = {token: idx for idx, token in enumerate(input_vocab_tokens)}

# Define the output (target) vocabulary.
# Output characters: digits 0-9 and ":".
target_vocab = {ch: idx for idx, ch in enumerate("0123456789:")}
output_vocab_size = len(target_vocab)  # Should be 11.

# Generate the dataset.
images, time_strings = generate_dataset(n=n, size=image_size)
print(f"Generated dataset with {len(images)} images.")

# Set up image transformation.
transform = transforms.Compose([
    transforms.ToTensor(),  # Scales image to [0, 1].
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the dataset.
dataset = ClockVLMDataset(images, time_strings, prompt, input_vocab, target_vocab, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model hyperparameters for the text branch.
vocab_size_input = len(input_vocab)
embed_dim = 32
text_hidden_dim = 128

# Initialize the model.
model = SimpleVLMSeq2Seq(vocab_size_input, embed_dim, text_hidden_dim, output_vocab_size, output_seq_len)
model.to(device)

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device, output_vocab_size)
    token_acc, sample_outputs = evaluate_model(model, test_loader, device, target_vocab, output_seq_len)
    print(f"Epoch {epoch+1}/{num_epochs} -- Loss: {train_loss:.4f} -- Token Acc: {token_acc:.4f}")

# Save the final model.
save_path = f"models/clock_vlm_seq2seq_model_{n}_{image_size}_{num_epochs}_{batch_size}_{learning_rate}.pth"
torch.save(model.state_dict(), save_path)
print("Training complete. Model saved as 'clock_vlm_seq2seq_model.pth'")


# Test inference on a single sample.
sample_img_np, sample_time_str = generate_clock_image(2, 3, n=n, size=image_size)
sample_img = transform(Image.fromarray(sample_img_np)).unsqueeze(0).to(device)
sample_text_indices = torch.tensor(dataset.prompt_indices, dtype=torch.long).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(sample_img, sample_text_indices)  # Shape: (1, seq_len, vocab_size)
    preds = output.argmax(dim=2).squeeze(0).cpu().numpy()
inv_target_vocab = {v: k for k, v in target_vocab.items()}
predicted_time = "".join([inv_target_vocab[token] for token in preds])
print("Ground truth time for sample image:", sample_time_str)
print("Predicted time for sample image:", predicted_time)