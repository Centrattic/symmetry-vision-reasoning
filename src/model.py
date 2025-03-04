import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from transformers import pipeline



device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)


reward_model_name = "facebook/opt-2.7b"  # Change to a smaller model if needed
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, num_labels=1).to(device)

# Define Reward Function
def compute_reward(text_explanation):
    inputs = reward_tokenizer(text_explanation, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze()
    return reward.item()


config = PPOConfig(
    model_name="Salesforce/blip2-opt-2.7b",
    learning_rate=1e-5,
    batch_size=4,
    log_with="wandb",  # Set up Weights & Biases for logging
)

# Define PPO Trainer
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset["train"],
    tokenizer=processor.tokenizer,
)

# PPO Training Loop
for epoch in range(3):  # Train for 3 epochs
    for batch in dataset["train"]:
        query_images = batch["pixel_values"]  # ARC input images

        # Generate explanations
        response = model.generate(input_ids=query_images, max_length=50)

        # Compute rewards
        reward = compute_reward(processor.tokenizer.decode(response[0]))

        # Train PPO
        ppo_trainer.step(query_images, response, reward)

# Save the trained PPO model
model.save_pretrained("./arc_blip2_ppo")
processor.save_pretrained("./arc_blip2_ppo")


def solve_arc_task(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, text="What is the transformation?", return_tensors="pt").to(device)

    # Generate explanation
    output = model.generate(input_ids=inputs.pixel_values, max_length=50)
    explanation = processor.tokenizer.decode(output[0])

    return explanation

# Example:
print(solve_arc_task("path/to/new_arc_problem.png"))
