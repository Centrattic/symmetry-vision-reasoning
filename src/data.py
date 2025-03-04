import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys

def load_json_data(json_path):
    """Load data from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_visualization(json_path, output_path=None, separate_input_output=False, output_path2 = None):
    """Create visualizations from the JSON data."""
    # Load the JSON data

    
    if output_path and os.path.exists(output_path):
        print(f"Output file already exists at {output_path}. Skipping visualization.")
        return

    data = load_json_data(json_path)
    
    # Process each dataset (train and test)
    all_examples = []
    train_examples = []
    test_examples = []
    
    if 'train' in data:
        train_examples = data['train']
        all_examples.extend(train_examples)
    if 'test' in data:
        test_examples = data['test']
        all_examples.extend(test_examples)
    
    # Create color mapping
    # 0: black (background)
    # 1: yellow
    # 4: green
    # 8: yellow
    # Any other values will get different colors
    colors = ['black', 'yellow', 'blue', 'red', 'green', 'purple', 'cyan', 'magenta', 'yellow']
    cmap = ListedColormap(colors)
    
    num_examples = len(all_examples)
    
    # If `separate_input_output` is True, save the input and output images separately
    if separate_input_output:
        # Save all input images in one figure
        plt.figure(figsize=(max(3 * num_examples, 10), 3), facecolor='black')
        for i, example in enumerate(all_examples):
            plt.subplot(1, num_examples, i+1)
            input_grid = np.array(example['input'])
            plt.imshow(input_grid, cmap=cmap, vmin=0, vmax=len(colors)-1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(color='gray', linestyle='-', linewidth=0.5)
            
            # Add white border
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

        # Save all output images in one figure
        plt.figure(figsize=(max(3 * num_examples, 10), 3), facecolor='black')
        for i, example in enumerate(all_examples):
            plt.subplot(1, num_examples, i+1)
            input_grid = np.array(example['input'])
            
            # For test examples, replace output with black grid
            is_test_example = example in test_examples
            if is_test_example:
                # Create a black grid of the same shape as input
                output_grid = np.zeros_like(input_grid)
            else:
                output_grid = np.array(example['output'])
            
            plt.imshow(output_grid, cmap=cmap, vmin=0, vmax=len(colors)-1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(color='gray', linestyle='-', linewidth=0.5)
            
            # Add white border
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

        return data
    
    # if not separate_input_output

    for i, example in enumerate(all_examples):
        fig, axes = plt.subplots(2, num_examples, figsize=(max(3 * num_examples, 10), 6), 
                           facecolor='black')
    
        # Handle case with only one example
        if num_examples == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        input_grid = np.array(example['input'])
        
        # For test examples, replace output with black grid
        is_test_example = example in test_examples
        if is_test_example:
            # Create a black grid of the same shape as input
            output_grid = np.zeros_like(input_grid)
        else:
            output_grid = np.array(example['output'])
        
        # Plot input grid
        axes[0, i].imshow(input_grid, cmap=cmap, vmin=0, vmax=len(colors)-1)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].grid(color='gray', linestyle='-', linewidth=0.5)
        
        # Add white border around the grid
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        
        # Plot output grid
        axes[1, i].imshow(output_grid, cmap=cmap, vmin=0, vmax=len(colors)-1)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].grid(color='gray', linestyle='-', linewidth=0.5)
        
        # Add white border around the grid
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        # Save or display the figure
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        else:
            plt.show()

    # Close the figure to free memory
    plt.close(fig)
    
    return data


def process_files(input_dir, output_dir, separate_input_output=False, output_dir2 = None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    output_path2 = None
    
    # Process files from directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, f"{os.path.basename(input_dir)}{file[1:-5]}.png")
            if separate_input_output and output_dir2:
                output_path2 = os.path.join(output_dir2, f"{os.path.basename(input_dir)}{file[1:-5]}.png")
            
            # Get and save visualization
            _ = create_visualization(input_path, output_path, separate_input_output, output_path2)
            print(f"Processed: {input_path} → {output_path}")

if __name__ == "__main__":

    # process_files("../ARC-AGI/data/evaluation/", "../img_data/evaluation/")
    # process_files("../ARC-AGI/data/training/", "../img_data/training/")

    process_files("../ARC-AGI/data/training/", output_dir = "../separate_img_data/training/input/", separate_input_output=True, output_dir2 = "../separate_img_data/training/output/")


