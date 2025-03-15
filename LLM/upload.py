from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi

dataset = load_from_disk("data")

# Push to Hugging Face Hub
dataset.push_to_hub("tinjet11/MedicalChatbot")