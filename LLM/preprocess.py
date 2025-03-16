from datasets import load_dataset, Dataset, concatenate_datasets
import json

# File size Too big(4.7gb), to be incorporate after poc
{"name": "ruslanmv/ai-medical-dataset", "split": "train", "columns": ["question", "context"]}, 

# To be add for more
datasets_info = [
    {"name": "medalpaca/medical_meadow_medical_flashcards", "split": "train", "columns": ["input", "output"]},
    {"name": "keivalya/MedQuad-MedicalQnADataset", "split": "train", "columns": ["Question", "Answer"]}
]

def rename_column(info, dataset): 
    match info["name"]:
        case "medalpaca/medical_meadow_medical_flashcards":
            dataset = dataset.rename_columns({"input": "question", "output": "answer"})  # Standardizing column names
            print("case 1")
        case "ruslanmv/ai-medical-dataset":
            dataset = dataset.rename_columns({"context": "answer"})  # Standardizing column names
            print("case 2")
        case "keivalya/MedQuad-MedicalQnADataset":
            dataset = dataset.rename_columns({"Question": "question", "Answer": "answer"})  # Standardizing column names
            print("case 3")
        case _:
            print("Default")
    return dataset

def convert_dataset_format(dataset):
    # Convert to JSONL format
    with open("train.jsonl", "w") as f:
        for ds in dataset["train"]:
            json.dump({"prompt": ds["question"], "response": ds["answer"]}, f)
            f.write("\n")  # Newline for JSONL format

    print("✅ Dataset converted to train.jsonl")


def combine_dataset(dataset):
    # Load datasets and unify structure
    datasets = []
    for info in datasets_info:
        dataset = load_dataset(info["name"], split=info["split"])
        # Rename or select columns to match a unified structure
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in info["columns"]])
        
        # Convert dataset to match unified schema
        dataset = rename_column(info,dataset) # Standardizing column names
        
        datasets.append(dataset)

    # Concatenate datasets
    combined_dataset = concatenate_datasets(datasets)

    # Display merged dataset
    print(combined_dataset)

    # Save to disk
    combined_dataset.save_to_disk("combined_dataset")


    