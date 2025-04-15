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


###############################
# 1. Data Generation
###############################

def generate_clock_image(hour, minute, n=5, size=128):
    """
    Generate a clock image with discrete positions.
    
    Args:
        hour (int): The hour-hand index (0 to n-1).
        minute (int): The minute-hand index (0 to n-1). The minute value will be computed as minute*(60/n)
        n (int): Number of discrete positions on the clock.
        size (int): The size (width and height) to which the image is resized.
    
    Returns:
        np.array: The generated RGB image as a numpy array.
        str: The time label as a string formatted "H:MM".
    """
    # Create a figure with no frame.
    fig, ax = plt.subplots(figsize=(2,2))
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.axis('off')
    
    # Draw the clock circle.
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_artist(circle)
    
    # Draw tick marks and number labels.
    for i in range(n):
        # Compute the angle so that index 0 is at the top.
        angle = math.pi/2 - 2 * math.pi * i / n
        x = 0.85 * math.cos(angle)
        y = 0.85 * math.sin(angle)
        ax.text(x, y, str(i), horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    # Draw the hour hand (shorter)
    hour_angle = math.pi/2 - 2 * math.pi * hour / n
    hour_x = 0.5 * math.cos(hour_angle)
    hour_y = 0.5 * math.sin(hour_angle)
    ax.plot([0, hour_x], [0, hour_y], color='black', linewidth=4)
    
    # Draw the minute hand (longer)
    minute_angle = math.pi/2 - 2 * math.pi * minute / n
    min_x = 0.8 * math.cos(minute_angle)
    min_y = 0.8 * math.sin(minute_angle)
    ax.plot([0, min_x], [0, min_y], color='red', linewidth=2)
    
    # Save the figure to a buffer.
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # Open the image with PIL and resize.
    img = Image.open(buf).convert('RGB')
    img = img.resize((size, size))
    
    # Compute the time label.
    # The minute value is computed by splitting the hour into n equal parts.
    minute_value = int(round(minute * (60 / n)))
    # Format minute to be two digits.
    time_label = f"{hour}:{minute_value:02d}"
    
    return np.array(img), time_label

def generate_dataset(n=5, size=128):
    """
    Generate a dataset of clock images and their labels.
    For each combination of hour (0 to n-1) and minute (0 to n-1), we generate an image.
    
    Args:
        n (int): Number of discrete clock positions.
        size (int): Image size (both height and width).
    
    Returns:
        images (list of np.array): The list of generated images.
        labels (list of int): The label for each image as an integer (hour * n + minute).
        time_strings (list of str): The human-readable time string corresponding to each image.
    """
    images = []
    labels = []
    time_strings = []
    for hour in range(n):
        for minute in range(n):
            img, time_str = generate_clock_image(hour, minute, n=n, size=size)
            images.append(img)
            # Use a single integer label: encode as hour * n + minute.
            label = hour * n + minute
            labels.append(label)
            time_strings.append(time_str)
    return images, labels, time_strings

###############################
# 2. Dataset Class for PyTorch
###############################

class ClockDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        A simple PyTorch Dataset wrapping clock images.
        
        Args:
            images (list of np.array): List of images.
            labels (list of int): Corresponding integer labels.
            transform: Optional torchvision transforms to apply.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label