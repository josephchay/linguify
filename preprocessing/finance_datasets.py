"""
Finance Dataset Processor
-------------------------
This script processes multiple finance-related datasets from Hugging Face,
transforms them into a standardized question-answer format, and saves them
to disk and optionally uploads them to the Hugging Face Hub.
"""

import json
import logging
from typing import Dict, List, Optional, Union, Any

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk
)
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASETS_CONFIG = [
    {
        "name": "gbharti/finance-alpaca",
        "split": "train",
        "columns": ["instruction", "input", "output"],
        "transform_type": "instruction"
    },
    {
        "name": "PaulAdversarial/all_news_finance_sm_1h2023",
        "split": "train",
        "columns": ["title", "description"],
        "transform_type": "news"
    },
    {
        "name": "winddude/reddit_finance_43_250k",
        "split": "train",
        "columns": ["title", "selftext", "body"],
        "transform_type": "reddit"
    },
    {
        "name": "causal-lm/finance",
        "split": "train",
        "columns": ["instruction", "input", "output"],
        "transform_type": "instruction"
    },
    {
        "name": "causal-lm/finance",
        "split": "validation",
        "columns": ["instruction", "input", "output"],
        "transform_type": "instruction"
    }
]


class FinanceDatasetProcessor:
    """Process finance datasets into a standardized question-answer format."""

    def __init__(self, config: List[Dict[str, Any]], output_dir: str = "finance_datasets"):
        self.datasets_config = config
        self.output_dir = output_dir
        self.combined_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def transform_instruction_dataset(self, example: Dict[str, Any]) -> Dict[str, str]:
        question = example["instruction"]
        if example.get("input") and example["input"].strip():
            question += f"\n\n{example['input']}"
        answer = example["output"]
        return {"question": question, "answer": answer}

    def transform_news_dataset(self, example: Dict[str, Any]) -> Dict[str, str]:
        question = f"Summarize the following financial news article: {example['title']}"
        answer = example["description"]
        return {"question": question, "answer": answer}

    def transform_reddit_dataset(self, example: Dict[str, Any]) -> Dict[str, str]:
        question = f"Financial Discussion: {example['title']}"
        answer = example["selftext"] if example.get("selftext") else ""
        if example.get("body") and example["body"].strip():
            if answer and answer.strip():
                answer += f"\n\n{example['body']}"
            else:
                answer = example["body"]
        return {"question": question, "answer": answer}

    def process_dataset(self, config: Dict[str, Any]) -> Dataset:
        logger.info(f"Processing {config['name']} ({config['split']} split)")

        # Load dataset
        dataset = load_dataset(config["name"], split=config["split"])

        # Apply transformation based on type
        if config["transform_type"] == "instruction":
            transformed_dataset = dataset.map(
                self.transform_instruction_dataset,
                remove_columns=config.get("columns", [])
            )
        elif config["transform_type"] == "news":
            transformed_dataset = dataset.map(
                self.transform_news_dataset,
                remove_columns=[col for col in dataset.column_names if col not in ["question", "answer"]]
            )
        elif config["transform_type"] == "reddit":
            transformed_dataset = dataset.map(
                self.transform_reddit_dataset,
                remove_columns=[col for col in dataset.column_names if col not in ["question", "answer"]]
            )
        else:
            raise ValueError(f"Unknown transform type: {config['transform_type']}")

        return transformed_dataset

    def process_all_datasets(self) -> Dataset:
        processed_datasets = []

        for config in self.datasets_config:
            dataset = self.process_dataset(config)
            processed_datasets.append(dataset)

        # Concatenate all datasets
        self.combined_dataset = concatenate_datasets(processed_datasets)
        logger.info(f"Combined dataset created with {len(self.combined_dataset)} examples")

        return self.combined_dataset

    def create_train_test_split(self, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
        if self.combined_dataset is None:
            raise ValueError("No combined dataset available. Run process_all_datasets first.")

        splits = self.combined_dataset.train_test_split(test_size=test_size, seed=seed)
        self.train_dataset = splits["train"]
        self.test_dataset = splits["test"]

        logger.info(f"Created train split with {len(self.train_dataset)} examples")
        logger.info(f"Created test split with {len(self.test_dataset)} examples")

        return splits

    def save_to_disk(self, base_path: Optional[str] = None) -> None:
        if base_path is None:
            base_path = self.output_dir

        if self.combined_dataset is not None:
            path = f"{base_path}/combined"
            self.combined_dataset.save_to_disk(path)
            logger.info(f"Combined dataset saved to {path}")

        if self.train_dataset is not None:
            path = f"{base_path}/train"
            self.train_dataset.save_to_disk(path)
            logger.info(f"Train dataset saved to {path}")

        if self.test_dataset is not None:
            path = f"{base_path}/test"
            self.test_dataset.save_to_disk(path)
            logger.info(f"Test dataset saved to {path}")

    def export_to_jsonl(self, base_path: Optional[str] = None) -> None:
        if base_path is None:
            base_path = self.output_dir

        def _write_jsonl(dataset: Dataset, filename: str):
            with open(filename, "w", encoding="utf-8") as f:
                for example in dataset:
                    json.dump({
                        "prompt": example["question"],
                        "response": example["answer"]
                    }, f, ensure_ascii=False)
                    f.write("\n")
            logger.info(f"Dataset exported to {filename}")

        if self.combined_dataset is not None:
            _write_jsonl(self.combined_dataset, f"{base_path}/combined.jsonl")

        if self.train_dataset is not None:
            _write_jsonl(self.train_dataset, f"{base_path}/train.jsonl")

        if self.test_dataset is not None:
            _write_jsonl(self.test_dataset, f"{base_path}/test.jsonl")

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        # Create a dataset dictionary to upload
        datasets_dict = DatasetDict()

        if self.combined_dataset is not None:
            datasets_dict["combined"] = self.combined_dataset

        if self.train_dataset is not None:
            datasets_dict["train"] = self.train_dataset

        if self.test_dataset is not None:
            datasets_dict["test"] = self.test_dataset

        # Push to Hub
        datasets_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=None  # Will use the token from huggingface-cli login
        )

        logger.info(f"All datasets pushed to Hugging Face Hub: {repo_id}")


def main():
    processor = FinanceDatasetProcessor(DATASETS_CONFIG)
    processor.process_all_datasets()
    processor.create_train_test_split(test_size=0.2)
    processor.save_to_disk()
    processor.export_to_jsonl()

    processor.push_to_hub("josephchay/LinguifyTTSDatasets")

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
