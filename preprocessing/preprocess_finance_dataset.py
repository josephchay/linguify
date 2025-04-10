"""
Finance Dataset Processor
-------------------------
This script processes multiple finance-related datasets from Hugging Face,
transforms them into a standardized question-answer format, and saves them
to disk and optionally uploads them to the Hugging Face Hub.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Callable

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets
)

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

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def transform_instruction_dataset(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Transform instruction-based datasets."""
        question = example.get("instruction", "")
        if not question:
            logger.warning("Missing instruction field in example")

        if example.get("input") and example["input"].strip():
            question += f"\n\n{example['input']}"

        answer = example.get("output", "")
        if not answer:
            logger.warning("Missing output field in example")

        return {"question": question, "answer": answer}

    def transform_news_dataset(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Transform news datasets."""
        title = example.get("title", "")
        if not title:
            logger.warning("Missing title field in example")
            title = "Untitled Financial News"

        question = f"Summarize the following financial news article: {title}"

        answer = example.get("description", "")
        if not answer:
            logger.warning("Missing description field in example")

        return {"question": question, "answer": answer}

    def transform_reddit_dataset(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Transform Reddit datasets."""
        title = example.get("title", "")
        if not title:
            logger.warning("Missing title field in example")
            title = "Untitled Financial Discussion"

        question = f"Financial Discussion: {title}"

        selftext = example.get("selftext", "")
        body = example.get("body", "")

        if selftext and selftext.strip():
            answer = selftext
            if body and body.strip():
                answer += f"\n\n{body}"
        else:
            answer = body if body else ""

        return {"question": question, "answer": answer}

    def get_transform_function(self, transform_type: str) -> Callable:
        """Get the appropriate transform function based on type."""
        transform_map = {
            "instruction": self.transform_instruction_dataset,
            "news": self.transform_news_dataset,
            "reddit": self.transform_reddit_dataset
        }

        if transform_type not in transform_map:
            raise ValueError(f"Unknown transform type: {transform_type}")

        return transform_map[transform_type]

    def process_dataset(self, config: Dict[str, Any]) -> Dataset:
        """Process a single dataset according to its configuration."""
        logger.info(f"Processing {config['name']} ({config['split']} split)")

        try:
            # Load dataset
            dataset = load_dataset(config["name"], split=config["split"])

            # Get transform function
            transform_func = self.get_transform_function(config["transform_type"])

            # Apply transformation - consistent column removal logic
            transformed_dataset = dataset.map(
                transform_func,
                remove_columns=[col for col in dataset.column_names if col not in ["question", "answer"]]
            )

            return transformed_dataset

        except Exception as e:
            logger.error(f"Error processing dataset {config['name']}: {str(e)}")
            raise

    def process_all_datasets(self) -> Dataset:
        """Process all configured datasets and combine them."""
        processed_datasets = []

        for config in self.datasets_config:
            try:
                dataset = self.process_dataset(config)
                processed_datasets.append(dataset)
            except Exception as e:
                logger.error(f"Skipping dataset {config['name']} due to error: {str(e)}")

        if not processed_datasets:
            raise ValueError("No datasets were successfully processed")

        # Concatenate all datasets
        self.combined_dataset = concatenate_datasets(processed_datasets)
        logger.info(f"Combined dataset created with {len(self.combined_dataset)} examples")

        return self.combined_dataset

    def create_train_test_split(self, test_size: float = 0.2, seed: int = None) -> DatasetDict:
        """Split the combined dataset into train and test sets."""
        if self.combined_dataset is None:
            raise ValueError("No combined dataset available. Run process_all_datasets first.")

        if not (0 < test_size < 1):
            raise ValueError(f"Invalid test_size: {test_size}. Must be between 0 and 1.")

        splits = self.combined_dataset.train_test_split(test_size=test_size, seed=seed)
        self.train_dataset = splits["train"]
        self.test_dataset = splits["test"]

        logger.info(f"Created train split with {len(self.train_dataset)} examples")
        logger.info(f"Created test split with {len(self.test_dataset)} examples")

        return splits

    def save_to_disk(self, base_path: Optional[str] = None) -> None:
        """Save datasets to disk."""
        if base_path is None:
            base_path = self.output_dir

        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)

        try:
            if self.combined_dataset is not None:
                path = os.path.join(base_path, "combined")
                self.combined_dataset.save_to_disk(path)
                logger.info(f"Combined dataset saved to {path}")

            if self.train_dataset is not None:
                path = os.path.join(base_path, "train")
                self.train_dataset.save_to_disk(path)
                logger.info(f"Train dataset saved to {path}")

            if self.test_dataset is not None:
                path = os.path.join(base_path, "test")
                self.test_dataset.save_to_disk(path)
                logger.info(f"Test dataset saved to {path}")
        except Exception as e:
            logger.error(f"Error saving datasets to disk: {str(e)}")
            raise

    def write_jsonl(self, dataset: Dataset, filename: str) -> None:
        """Write a dataset to a JSONL file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for example in dataset:
                    json.dump({
                        "prompt": example["question"],
                        "response": example["answer"]
                    }, f, ensure_ascii=False)
                    f.write("\n")
            logger.info(f"Dataset exported to {filename}")
        except Exception as e:
            logger.error(f"Error writing JSONL file {filename}: {str(e)}")
            raise

    def export_to_jsonl(self, base_path: Optional[str] = None) -> None:
        """Export datasets to JSONL format."""
        if base_path is None:
            base_path = self.output_dir

        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)

        if self.combined_dataset is not None:
            self.write_jsonl(self.combined_dataset, os.path.join(base_path, "combined.jsonl"))

        if self.train_dataset is not None:
            self.write_jsonl(self.train_dataset, os.path.join(base_path, "train.jsonl"))

        if self.test_dataset is not None:
            self.write_jsonl(self.test_dataset, os.path.join(base_path, "test.jsonl"))

    def push_to_hub(self, repo_id: str, private: bool = False, token: Optional[str] = None) -> None:
        """Push datasets to Hugging Face Hub."""
        if not repo_id:
            raise ValueError("Repository ID is required")

        # Create a dataset dictionary to upload
        datasets_dict = DatasetDict()

        if self.combined_dataset is not None:
            datasets_dict["combined"] = self.combined_dataset

        if self.train_dataset is not None:
            datasets_dict["train"] = self.train_dataset

        if self.test_dataset is not None:
            datasets_dict["test"] = self.test_dataset

        if not datasets_dict:
            raise ValueError("No datasets available to push to Hub")

        try:
            # Push to Hub
            datasets_dict.push_to_hub(
                repo_id=repo_id,
                private=private,
                token=token  # Will use the token from huggingface-cli login if None
            )
            logger.info(f"All datasets pushed to Hugging Face Hub: {repo_id}")
        except Exception as e:
            logger.error(f"Error pushing to Hugging Face Hub: {str(e)}")
            raise


def main():
    try:
        output_dir = "datasets/finance"
        test_size = 0.2  # set to 0.2 for 80-20 split
        seed = None  # used if want to reproduce the same split / put None if you want random split
        repo_id = "josephchay/LinguifyDataset"

        processor = FinanceDatasetProcessor(DATASETS_CONFIG, output_dir=output_dir)
        processor.process_all_datasets()
        processor.create_train_test_split(test_size=test_size, seed=seed)
        processor.save_to_disk()
        processor.export_to_jsonl()
        processor.push_to_hub(repo_id)

        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
