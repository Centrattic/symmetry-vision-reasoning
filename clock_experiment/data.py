import os
import math
import random
import numpy as np
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset

import torch


def generate_clock_image(hour, minute, n=5, size=128):
    """
    Generate a synthetic clock image with n discrete positions.
    
    The clock image is drawn with matplotlib. The time is computed as:
      - hour: as provided (0 to n-1)
      - minute_value: computed as int(round(minute * (60/n)))
      
    The time string is formatted as hh:mm with leading zeros.
    
    Returns:
      np.array: the generated RGB image.
      str: the corresponding time string (e.g., "02:30").
    """
    fig, ax = plt.subplots(figsize=(2,2))
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.axis('off')
    
    # Draw clock circle.
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_artist(circle)
    
    # Draw tick marks and number labels.
    for i in range(n):
        angle = math.pi/2 - 2 * math.pi * i / n
        x = 0.85 * math.cos(angle)
        y = 0.85 * math.sin(angle)
        ax.text(x, y, str(i), horizontalalignment='center',
                verticalalignment='center', fontsize=12)
    
    # Draw hour hand (shorter).
    hour_angle = math.pi/2 - 2 * math.pi * hour / n
    hour_x = 0.5 * math.cos(hour_angle)
    hour_y = 0.5 * math.sin(hour_angle)
    ax.plot([0, hour_x], [0, hour_y], color='black', linewidth=4)
    
    # Draw minute hand (longer).
    minute_angle = math.pi/2 - 2 * math.pi * minute / n
    min_x = 0.8 * math.cos(minute_angle)
    min_y = 0.8 * math.sin(minute_angle)
    ax.plot([0, min_x], [0, min_y], color='red', linewidth=2)
    
    # Save the figure to a buffer.
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # Open with PIL and resize.
    img = Image.open(buf).convert('RGB')
    img = img.resize((size, size))
    
    minute_value = int(round(minute * (60 / n)))
    time_str = f"{hour:02d}:{minute_value:02d}"
    return np.array(img), time_str

def generate_dataset(n=5, size=128):
    """
    Generate a dataset of clock images.
    
    For every combination of hour (0 to n-1) and minute (0 to n-1) an image is generated.
    
    Returns:
      images: list of np.array images.
      time_strings: list of corresponding time strings in the format hh:mm.
    """
    images = []
    time_strings = []
    for hour in range(n):
        for minute in range(n):
            img, time_str = generate_clock_image(hour, minute, n=n, size=size)
            images.append(img)
            time_strings.append(time_str)
    return images, time_strings

def time_str_to_indices(time_str, out_vocab):
    """
    Convert a time string (e.g., "04:20") to a list of token indices using out_vocab.
    Assumes that the time string always has length 5 (hh:mm).
    """
    return [out_vocab[char] for char in time_str]

def tokenize(text):
    """Simple whitespace tokenizer and lowercase."""
    return text.lower().split()

def text_to_indices(text, vocab):
    """Convert a text string to a list of token indices based on the provided vocab."""
    tokens = tokenize(text)
    return [vocab[token] for token in tokens]

###############################
# 3. Vision-Language Dataset Class
###############################

class ClockVLMDataset(Dataset):
    def __init__(self, images, time_strings, prompt, input_vocab, target_vocab, transform=None):
        """
        Dataset returns:
          image, input prompt tokens, and target output sequence tokens.
          
        Args:
          images: list of np.array images.
          time_strings: list of time strings (target labels), each of length 5 (hh:mm).
          prompt: a fixed text prompt (e.g., "Tell me the time on the clock in hh:mm format").
          input_vocab: vocabulary for tokenizing the prompt.
          target_vocab: vocabulary for converting the time string to indices.
          transform: image transformation (e.g., converting to Tensor).
        """
        self.images = images
        self.time_strings = time_strings
        self.prompt = prompt
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.transform = transform
        # Precompute prompt token indices.
        self.prompt_indices = torch.tensor(self.text_to_indices(prompt, input_vocab), dtype=torch.long)
        # Precompute target sequences (each a list of 5 token indices).
        self.target_seqs = [torch.tensor(time_str_to_indices(ts, target_vocab), dtype=torch.long) 
                            for ts in self.time_strings]
    
    def text_to_indices(self, text, vocab):
        # Simple whitespace tokenizer.
        tokens = text.lower().split()
        return [vocab[token] for token in tokens if token in vocab]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        target_seq = self.target_seqs[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.prompt_indices, target_seq
