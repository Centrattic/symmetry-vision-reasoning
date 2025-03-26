import torch
from PIL import Image
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor, CLIPVisionModel
from trl import SmolPPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import DataLoader

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VISION_ENCODER = "openai/clip-vit-base-patch32"
TEXT_MODEL = "teknium/SmolLLaVA-1.5-1.3B"

# Load ARC-AGI dataset
class ARCDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = load_dataset("allenai/arc", "ARC-AGI")
        
    def __getitem__(self, idx):
        item = self.data["train"][idx]  # Explicitly selecting the split
        return {
            "image_path": item["image"],
            "question": item["question"],
            "answer": item["answer"]
        }

    def __len__(self):
        return len(self.data["train"])

# Reward function for visual reasoning
def visual_reward_function(outputs, answers):
    rewards = [1.0 if output.strip() == answer.strip() else 0.0 for output, answer in zip(outputs, answers)]
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

# Initialize VLM components
def initialize_vlm():
    vision_encoder = CLIPVisionModel.from_pretrained(VISION_ENCODER).to(DEVICE)
    processor = AutoProcessor.from_pretrained(TEXT_MODEL)
    model = LlavaForConditionalGeneration.from_pretrained(TEXT_MODEL, torch_dtype=torch.float16).to(DEVICE)
    model = AutoModelForCausalLMWithValueHead(model)  # Add value head for PPO
    return vision_encoder, processor, model

# Main training loop
def main():
    vision_encoder, processor, model = initialize_vlm()
    dataset = ARCDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    ppo_config = PPOConfig(batch_size=4, mini_batch_size=2, learning_rate=1e-5)
    ppo_trainer = SmolPPOTrainer(model=model, config=ppo_config, tokenizer=processor.tokenizer)
    
    for epoch in range(10):
        for batch in dataloader:
            images = [Image.open(img_path) for img_path in batch["image_path"]]
            inputs = processor(text=batch["question"], images=images, return_tensors="pt", padding=True).to(DEVICE)
            
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.95)
            decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
            
            rewards = visual_reward_function(decoded_outputs, batch["answer"])
            ppo_trainer.step(queries=batch["question"], responses=decoded_outputs, rewards=rewards)
        
        print(f"Epoch {epoch + 1} completed.")
    
    model.save_pretrained("arc-vlm-ppo")
    processor.save_pretrained("arc-vlm-ppo")

if __name__ == "__main__":
    main()
