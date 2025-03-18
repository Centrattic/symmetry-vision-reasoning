import torch
from transformers import AutoModel, AutoTokenizer
import os
from glob import glob
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def process_images(model, tokenizer, input_folder, output_filepath, query):
    """
    Process all images in the input folder and save results to the output folder.
    
    Args:
        model: The model to use for inference
        tokenizer: The tokenizer for the model
        input_folder: Path to folder containing images
        output_folder: Path to save results
        query: Query to use with each image
    """
    
    # Get all image files in the input folder
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    
    for format in supported_formats:
        image_files.extend(glob(os.path.join(input_folder, format)))
    
    print(f"Found {len(image_files)} images to process")

    # Create output file path
    base_filename = input_folder.split(os.sep)[-1] # the evaluation/training folder
    # output_path = output_filepath + base_filename + ".txt"
    
    # Process each image
    for image_path in image_files:
        output_path = output_filepath + base_filename + image_path.split(os.sep)[-1][:-4] + ".txt"
        print(f"Processing: {image_path}")
        # try:
        # Load the image
        image = [image_path]
        pixel_values = load_image(image[0], max_num=100).to(torch.bfloat16).cuda()
        
        # Run the model with the same settings as your example
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            response = model.chat(tokenizer, pixel_values, query, generation_config)
        
        # Save the results
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Image: {str(image_path)}\n\n")
            f.write(f"Response:\n{str(response)}")
        
        print(f"Saved results to: {str(output_path)}")
            
        # except Exception as e:
            # print(f"Error processing {image_path}: {str(e)}")
    
    print("Processing complete!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images through the model')
    parser.add_argument('--input', required=True, help='Input folder containing images')
    parser.add_argument('--type', default='', help='Model type')
    parser.add_argument('--model', default='', help='Model name')
    
    args = parser.parse_args()

    model_type = args.type
    model_name = args.model

    input_path = f"../img_data/{args.input}/"

    model_path_name = model_name.split(os.sep)[-1]
    output_path = f"../results/{model_type}_{model_path_name}_"

    # if not os.path.exists(output_path):
    # # do not overwrite
    #     os.mkdir(output_path)

    torch.set_grad_enabled(False)

    # init model and tokenizer
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.tokenizer = tokenizer

    # In addition to the image, I will give you an array representation for all input and output grids seen. This grid is
            #  a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 
        #     1x1 and the largest is 30x30.

    query = """This image is a logic puzzle that contains input and output examples. Your job is to learn
                the transformation between the input and output grids and apply it to the final grid, which is empty (a black box).Please construct the final output grid by picking the height and width, then filling each 
                cell in the grid with a symbol (integer between 0 and 9, visualized as colors). Only exact solutions 
                are correct. Please output an array representing the output grid."""
    
    print("Processing time")
    
    # Process the images
    process_images(model, tokenizer, input_path, output_path, query)

    #   model_type = "visual-qa"
    #   model_name = "internlm/internlm-xcomposer2d5-7b"

if __name__ == "__main__":
    main()


# python vqa_run.py --input training --type visual-qa --model internlm/internlm-xcomposer2d5-7b

# python vqa_run.py --input training --type visual-qa --model openbmb/MiniCPM-V
# python vqa_run.py --input training --type visual-qa --model Salesforce/blip2-opt-2.7b
# python vqa_run.py --input training --type visual-qa --model OpenGVLab/InternVL2-1B
# python vqa_run.py --input training --type visual-qa --model antphb/Vinqw-1B-v1
# python vqa_run.py --input training --type visual-qa --model mPLUG/mPLUG-Owl3-1B-241014
 

