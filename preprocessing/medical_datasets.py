"""
Medical Dataset Processor
------------------------
This script processes multiple medical-related datasets from Hugging Face,
transforms them into a standardized question-answer format, and saves them
to disk and optionally uploads them to the Hugging Face Hub.
"""

import json
import re
import logging
import os

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk
)
from huggingface_hub import HfApi


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATASETS_CONFIG = [
    {
        "name": "medalpaca/medical_meadow_medical_flashcards",
        "split": "train",
        "rename_map": {"input": "question", "output": "answer"}
    },
    {
        "name": "keivalya/MedQuad-MedicalQnADataset",
        "split": "train",
        "rename_map": {"Question": "question", "Answer": "answer"}
    },
    {
        "name": "bio-nlp-umass/bioinstruct",
        "split": "train",
        "transform": lambda ex: {
            "question": f"{ex['instruction']}: {ex['input']}",
            "answer": ex['output']
        },
        "remove_columns": ["instruction", "input", "output"]
    },
    {
        "name": "qiaojin/PubMedQA",
        "split": "train",
        "config": "pqa_artificial",
        "transform": lambda ex: {
            "question": f"Question: {ex['question']}\n\nContext: {' '.join(ex['context'].get('contexts', []))}\n\nLabels: {', '.join(ex['context'].get('labels', []))}\n\nMeSH Terms: {', '.join(ex['context'].get('meshes', []))}",
            "answer": f"{ex.get('long_answer', '')}\n\nFinal Decision: {ex.get('final_decision', '')}" if ex.get('final_decision') else ex.get('long_answer', '')
        },
        "remove_columns": ["question", "context", "long_answer", "final_decision"]
    },
    {
        "name": "RafaelMPereira/HealthCareMagic-100k-Chat-Format-en",
        "split": "train",
        "transform": lambda ex: {
            "question": re.search(r"<human>:\s*(.*?)\s*<bot>:", ex["text"], re.DOTALL).group(1).strip() if re.search(r"<human>:\s*(.*?)\s*<bot>:", ex["text"], re.DOTALL) else "No question found",
            "answer": re.search(r"<bot>:\s*(.*)", ex["text"], re.DOTALL).group(1).strip() if re.search(r"<bot>:\s*(.*)", ex["text"], re.DOTALL) else "No answer found"
        },
        "remove_columns": ["text"]
    },
    {
        "name": "ruslanmv/ai-medical-dataset",
        "split": "train",
        "rename_map": {"question": "question", "context": "answer"}
    },
    {
        "name": "Malikeh1375/medical-question-answering-datasets",
        "split": "train",
        "rename_map": {"input": "question", "output": "answer"}
    },
    {
        "name": "medalpaca/medical_meadow_wikidoc",
        "split": "train",
        "rename_map": {"input": "question", "output": "answer"}
    },
    {
        "name": "Mohammed-Altaf/medical-instruction-100k",
        "split": "train",
        "transform": lambda ex: {
            "question": re.search(r"\[|Human|\](.*?)\[|AI|\]", ex["Conversation"].replace("The conversation between human and AI assistant.", "", 1).strip(), re.DOTALL).group(1).strip()
                      if re.search(r"\[|Human|\](.*?)\[|AI|\]", ex["Conversation"].replace("The conversation between human and AI assistant.", "", 1).strip(), re.DOTALL)
                      else "No question found",
            "answer": re.search(r"\[|AI|\](.*?)$", ex["Conversation"].replace("The conversation between human and AI assistant.", "", 1).strip(), re.DOTALL).group(1).strip()
                     if re.search(r"\[|AI|\](.*?)$", ex["Conversation"].replace("The conversation between human and AI assistant.", "", 1).strip(), re.DOTALL)
                     else "No answer found"
        },
        "remove_columns": ["Conversation"]
    },
    {
        "name": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "split": "train",
        "rename_map": {"Question": "question", "Response": "answer"}
    },
    {
        "name": "lavita/medical-qa-datasets",
        "split": "train",
        "rename_map": {"input": "question", "output": "answer"}
    },
    {
        "name": "BI55/MedText",
        "split": "train",
        "rename_map": {"Prompt": "question", "Completion": "answer"}
    }
]

def load_and_process_dataset(config):
    logger.info(f"Processing {config['name']} ({config['split']} split)")

    try:
        if config.get("config"):
            dataset = load_dataset(config["name"], config["config"], split=config["split"])
        else:
            dataset = load_dataset(config["name"], split=config["split"])

        if config.get("rename_map"):
            dataset = dataset.rename_columns(config["rename_map"])

        if config.get("transform"):
            dataset = dataset.map(
                config["transform"],
                remove_columns=config.get("remove_columns", [])
            )

        keep_columns = ["question", "answer"]
        remove_columns = [col for col in dataset.column_names if col not in keep_columns]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

        logger.info(f"Successfully processed {config['name']} with {len(dataset)} examples")
        return dataset

    except Exception as e:
        logger.error(f"Error processing {config['name']}: {str(e)}")
        return None

def process_datasets(configs):
    processed_datasets = []

    for config in configs:
        dataset = load_and_process_dataset(config)
        if dataset:
            processed_datasets.append(dataset)

    if not processed_datasets:
        raise ValueError("No datasets were successfully processed")

    # Combine all processed datasets
    combined_dataset = concatenate_datasets(processed_datasets)
    logger.info(f"Combined dataset created with {len(combined_dataset)} examples")

    return combined_dataset

def save_dataset(dataset, output_dir="medical_datasets"):
    os.makedirs(output_dir, exist_ok=True)

    # Create train/test split
    splits = dataset.train_test_split(test_size=0.2)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    logger.info(f"Created train split with {len(train_dataset)} examples")
    logger.info(f"Created test split with {len(test_dataset)} examples")

    # Save combined dataset
    dataset.save_to_disk(f"{output_dir}/combined")
    logger.info(f"Combined dataset saved to {output_dir}/combined")

    # Save train and test splits
    train_dataset.save_to_disk(f"{output_dir}/train")
    test_dataset.save_to_disk(f"{output_dir}/test")
    logger.info(f"Train and test datasets saved to {output_dir}")

    # Export to JSONL format
    export_to_jsonl(dataset, f"{output_dir}/combined.jsonl")
    export_to_jsonl(train_dataset, f"{output_dir}/train.jsonl")
    export_to_jsonl(test_dataset, f"{output_dir}/test.jsonl")

    return {"combined": dataset, "train": train_dataset, "test": test_dataset}

def export_to_jsonl(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            json.dump({"prompt": example["question"], "response": example["answer"]}, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"Dataset exported to {filename}")

def push_to_hugging_face(datasets_dict, repo_id):
    hf_datasets = DatasetDict({
        name: dataset for name, dataset in datasets_dict.items()
    })

    hf_datasets.push_to_hub(repo_id)
    logger.info(f"Datasets pushed to Hugging Face Hub: {repo_id}")

def main():
    combined_dataset = process_datasets(DATASETS_CONFIG)
    datasets_dict = save_dataset(combined_dataset)

    push_to_hugging_face(datasets_dict, "josephchay/LinguifyTTSDatasets")

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
