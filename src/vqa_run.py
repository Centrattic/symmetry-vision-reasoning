import torch
from transformers import AutoModel, AutoTokenizer
import os
from glob import glob
import argparse

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
    output_path = output_filepath + base_filename + ".txt"
    
    # Process each image
    for image_path in image_files:
        print(f"Processing: {image_path}")
        try:
            # Load the image
            image = [image_path]
            
            # Run the model with the same settings as your example
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
            
            # Save the results
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n\n")
                f.write(f"Response:\n{response}")
            
            print(f"Saved results to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
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

    input_path = f"../ARC-AGI/data/{args.input}/"

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

    query = """This image is a logic puzzle that contains input and output examples. Your job is to learn
                the transformation between the input and output grids and apply it to the final grid, which has no associated output (appears as a black box).
                In addition to the image, I will give you an array representation for all input and output grids seen. This grid is
                a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 
                1x1 and the largest is 30x30.

                Your goal is to construct the output grid corresponding to the test input grid. 
                "Constructing the output grid" involves picking the height and width of the output grid, then filling each 
                cell in the grid with a symbol (integer between 0 and 9, which are visualized as colors). Only exact solutions 
                (all cells match the expected answer) can be said to be correct. Please output an array representing the output grid.
                If you are able to generate images, output an image visualization of the grid."""
    
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
 

