"""
Medical Dataset Processor
------------------------
This script processes multiple medical-related datasets from Hugging Face,
transforms them into a standardized question-answer format, and saves them
to disk and optionally uploads them to the Hugging Face Hub.

The script handles various dataset formats and ensures all datasets are
properly converted to a consistent question-answer format.
"""

import json
import re
import logging
import os
from typing import Dict, List, Optional, Any, Tuple

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define all transformation functions separately
def transform_bioinstruct(ex):
    return {
        "question": f"{ex['instruction']}: {ex['input']}",
        "answer": ex['output']
    }


def transform_pubmedqa(ex):
    return {
        "question": f"Question: {ex['question']}\n\nContext: {' '.join(ex['context'].get('contexts', []))}\n\nLabels: {', '.join(ex['context'].get('labels', []))}\n\nMeSH Terms: {', '.join(ex['context'].get('meshes', []))}",
        "answer": f"{ex.get('long_answer', '')}\n\nFinal Decision: {ex.get('final_decision', '')}" if ex.get(
            'final_decision') else ex.get('long_answer', '')
    }


def transform_healthcare_magic(ex):
    return {
        "question": re.search(r"<human>:\s*(.*?)\s*<bot>:", ex["text"], re.DOTALL).group(1).strip() if re.search(r"<human>:\s*(.*?)\s*<bot>:", ex["text"], re.DOTALL) else "No question found",
        "answer": re.search(r"<bot>:\s*(.*)", ex["text"], re.DOTALL).group(1).strip() if re.search(r"<bot>:\s*(.*)", ex["text"], re.DOTALL) else "No answer found"
    }


def transform_medical_instruction(ex):
    # Safe extraction that handles potential None values or missing patterns
    conversation = ex.get("Conversation", "")
    # Remove initial sentence if present
    if isinstance(conversation, str) and conversation.startswith("The conversation between human and AI assistant."):
        conversation = conversation.replace("The conversation between human and AI assistant.", "", 1).strip()

    # Safe pattern matching with fixed regex (escaped pipe characters)
    human_match = re.search(r"\[\|Human\|\](.*?)\[\|AI\|\]", conversation, re.DOTALL) if isinstance(conversation, str) else None
    ai_match = re.search(r"\[\|AI\|\](.*?)$", conversation, re.DOTALL) if isinstance(conversation, str) else None

    question = human_match.group(1).strip() if human_match else "No question found"
    answer = ai_match.group(1).strip() if ai_match else "No answer found"

    return {"question": question, "answer": answer}


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
        "transform": transform_bioinstruct,
        "remove_columns": ["instruction", "input", "output"]
    },
    {
        "name": "qiaojin/PubMedQA",
        "split": "train",
        "config": "pqa_artificial",
        "transform": transform_pubmedqa,
        "remove_columns": ["question", "context", "long_answer", "final_decision"]
    },
    {
        "name": "RafaelMPereira/HealthCareMagic-100k-Chat-Format-en",
        "split": "train",
        "transform": transform_healthcare_magic,
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
        "config": "all-processed",
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
        "transform": transform_medical_instruction,
        "remove_columns": ["Conversation"]
    },
    {
        "name": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "split": "train",
        "config": "en",
        "rename_map": {"Question": "question", "Response": "answer"}
    },
    {
        "name": "lavita/medical-qa-datasets",
        "split": "train",
        "config": "all-processed",
        "rename_map": {"input": "question", "output": "answer"}
    },
    {
        "name": "BI55/MedText",
        "split": "train",
        "rename_map": {"Prompt": "question", "Completion": "answer"}
    }
]


def load_and_process_dataset(config: Dict[str, Any]) -> Optional[Dataset]:
    logger.info(f"Processing {config['name']} ({config['split']} split)")

    try:
        # Load dataset with config if provided
        if config.get("config"):
            dataset = load_dataset(config["name"], config["config"], split=config["split"])
        else:
            dataset = load_dataset(config["name"], split=config["split"])

        # Apply simple column renaming if specified
        if config.get("rename_map"):
            # Only rename columns that actually exist in the dataset
            rename_dict = {k: v for k, v in config["rename_map"].items() if k in dataset.column_names}
            if rename_dict:
                dataset = dataset.rename_columns(rename_dict)

        # Apply custom transformation if specified
        if config.get("transform"):
            dataset = dataset.map(
                config["transform"],
                remove_columns=config.get("remove_columns", [])
            )

        # Ensure dataset has question and answer columns
        if "question" not in dataset.column_names or "answer" not in dataset.column_names:
            logger.warning(f"Dataset {config['name']} is missing required 'question' or 'answer' columns after processing")
            return None

        # Remove any columns other than question and answer
        columns_to_keep = ["question", "answer"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        # Validate data - ensure no None values or empty strings
        def validate_example(example):
            valid_question = isinstance(example["question"], str) and example["question"].strip()
            valid_answer = isinstance(example["answer"], str) and example["answer"].strip()
            return valid_question and valid_answer

        # Filter out invalid examples
        valid_examples = [validate_example(ex) for ex in dataset]
        if not all(valid_examples):
            invalid_count = valid_examples.count(False)
            logger.warning(f"Removing {invalid_count} invalid examples from {config['name']}")
            dataset = dataset.filter(lambda ex: isinstance(ex["question"], str) and
                                              isinstance(ex["answer"], str) and
                                              ex["question"].strip() and
                                              ex["answer"].strip())

        logger.info(f"Successfully processed {config['name']} with {len(dataset)} examples")
        return dataset

    except Exception as e:
        logger.error(f"Error processing {config['name']}: {str(e)}")
        return None

def process_datasets(configs: List[Dict[str, Any]], max_examples_per_dataset: Optional[int] = None) -> Dataset:
    processed_datasets = []

    for config in configs:
        dataset = load_and_process_dataset(config)
        if dataset:
            # Optionally limit dataset size (useful for testing)
            if max_examples_per_dataset and len(dataset) > max_examples_per_dataset:
                dataset = dataset.select(range(max_examples_per_dataset))
                logger.info(f"Limited {config['name']} to {max_examples_per_dataset} examples")

            processed_datasets.append(dataset)

    if not processed_datasets:
        raise ValueError("No datasets were successfully processed")

    # Combine all processed datasets
    combined_dataset = concatenate_datasets(processed_datasets)
    logger.info(f"Combined dataset created with {len(combined_dataset)} examples")

    return combined_dataset

def create_train_test_split(dataset: Dataset, test_size: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset]:
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    logger.info(f"Created train split with {len(train_dataset)} examples")
    logger.info(f"Created test split with {len(test_dataset)} examples")

    return train_dataset, test_dataset

def save_datasets(combined_dataset: Dataset, train_dataset: Dataset, test_dataset: Dataset, output_dir: str = "datasets/medical") -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets to disk
    combined_dataset.save_to_disk(f"{output_dir}/combined")
    logger.info(f"Combined dataset saved to {output_dir}/combined")

    train_dataset.save_to_disk(f"{output_dir}/train")
    logger.info(f"Train dataset saved to {output_dir}/train")

    test_dataset.save_to_disk(f"{output_dir}/test")
    logger.info(f"Test dataset saved to {output_dir}/test")

def export_to_jsonl(datasets: Dict[str, Dataset], output_dir: str = "medical_datasets") -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name, dataset in datasets.items():
        filename = f"{output_dir}/{name}.jsonl"

        with open(filename, "w", encoding="utf-8") as f:
            for i, example in enumerate(dataset):
                # Status update for large datasets
                if i % 1000000 == 0 and i > 0:
                    logger.info(f"Exported {i} examples to {filename}")

                json.dump({
                    "prompt": example["question"],
                    "response": example["answer"]
                }, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Dataset exported to {filename}")

def push_to_hugging_face(datasets: Dict[str, Dataset], repo_id: str, private: bool = False) -> None:
    try:
        # Create DatasetDict for uploading
        hf_datasets = DatasetDict({
            name: dataset for name, dataset in datasets.items()
        })

        # Push to Hub
        hf_datasets.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=None  # Will use the token from huggingface-cli login
        )
        logger.info(f"Datasets pushed to Hugging Face Hub: {repo_id}")
    except Exception as e:
        logger.error(f"Error pushing to Hugging Face Hub: {str(e)}")

def main():
    MAX_EXAMPLES_PER_DATASET = None

    try:
        combined_dataset = process_datasets(DATASETS_CONFIG, MAX_EXAMPLES_PER_DATASET)
        train_dataset, test_dataset = create_train_test_split(combined_dataset)
        datasets_dict = {
            "combined": combined_dataset,
            "train": train_dataset,
            "test": test_dataset
        }

        repo_id = "josephchay/LinguifyDataset"

        # Save datasets to disk
        save_datasets(combined_dataset, train_dataset, test_dataset)
        export_to_jsonl(datasets_dict)
        push_to_hugging_face(datasets_dict, repo_id)

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()
