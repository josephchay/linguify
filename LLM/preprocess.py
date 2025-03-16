from datasets import load_dataset, Dataset, concatenate_datasets
import json
import re

# Dataset sources
datasets_info = [
    {"name": "medalpaca/medical_meadow_medical_flashcards",
        "split": "train", "columns": ["input", "output"]},
    {"name": "keivalya/MedQuad-MedicalQnADataset",
        "split": "train", "columns": ["Question", "Answer"]},
    {"name": "bio-nlp-umass/bioinstruct", "split": "train",
        "columns": ["instruction", "input", "output"]},
    {"name": "qiaojin/PubMedQA", "split": "train",
        "columns": ["question", "context", "long_answer", "final_decision"]},
    {"name": "RafaelMPereira/HealthCareMagic-100k-Chat-Format-en",
        "split": "train", "columns": ["text"]},
    {"name": "ruslanmv/ai-medical-dataset", "split": "train",
        "columns": ["question", "context"]},
]

# Transform for bio-nlp-umass/bioinstruct
def transform1(example):
    question = f"{example['instruction']}: {example['input']}"
    answer = example['output']
    return {"question": question, "answer": answer}

# Transform for qiaojin/PubMedQA
def transform2(example):
    # Extract the question
    question = example["question"]

    # Extract context components
    contexts = " ".join(example["context"].get(
        "contexts", []))  # Merge all contexts
    labels = ", ".join(example["context"].get(
        "labels", []))  # Convert list to string
    meshes = ", ".join(example["context"].get(
        "meshes", []))  # Convert list to string

    # Build the full question with context, labels, and meshes
    full_question = f"Question: {question}\n\nContext: {contexts}\n\nLabels: {labels}\n\nMeSH Terms: {meshes}"

    # Extract the long answer and final decision
    long_answer = example.get("long_answer", "")
    final_decision = example.get("final_decision", "")

    # Build the full answer
    full_answer = f"{long_answer}\n\nFinal Decision: {final_decision}" if final_decision else long_answer

    return {"question": full_question, "answer": full_answer}

# Transform for RafaelMPereira/HealthCareMagic-100k-Chat-Format-en
def transform3(example):
    question_match = re.search(
        r"<human>:\s*(.*?)\s*<bot>:", example["text"], re.DOTALL)
    answer_match = re.search(r"<bot>:\s*(.*)", example["text"], re.DOTALL)

    question = question_match.group(1).strip(
    ) if question_match else "No question found"
    answer = answer_match.group(1).strip(
    ) if answer_match else "No answer found"

    return {"question": question, "answer": answer}


# Rename and format dataset columns
def rename_column(info, dataset):
    if info["name"] == "medalpaca/medical_meadow_medical_flashcards":
        dataset = dataset.rename_columns(
            {"input": "question", "output": "answer"})
    elif info["name"] == "keivalya/MedQuad-MedicalQnADataset":
        dataset = dataset.rename_columns(
            {"Question": "question", "Answer": "answer"})
    elif info["name"] == "bio-nlp-umass/bioinstruct":
        dataset = dataset.map(transform1, remove_columns=[
                              "instruction", "input", "output"])
    elif info["name"] == "qiaojin/PubMedQA":
        dataset = dataset.map(transform2, remove_columns=[
                              "question", "context", "long_answer", "final_decision"])
    elif info["name"] == "RafaelMPereira/HealthCareMagic-100k-Chat-Format-en":
        dataset = dataset.map(transform3, remove_columns=["text"])
    elif info["name"] == "ruslanmv/ai-medical-dataset":
        dataset = dataset.rename_columns(
            {"question": "question", "context": "answer"})
    return dataset

# Convert dataset to JSONL format
def convert_dataset_format(dataset, filename="train.jsonl"):
    with open(filename, "w") as f:
        for example in dataset:
            json.dump({"prompt": example["question"],
                      "response": example["answer"]}, f)
            f.write("\n")  # Newline for JSONL format
    print(f"✅ Dataset converted and saved as {filename}")

# Load, process, and combine datasets
def combine_dataset():
    datasets = []
    for info in datasets_info:
        print(info["name"])

        if (info["name"] == "qiaojin/PubMedQA"):
            dataset = load_dataset(
                info["name"], "pqa_artificial", split=info["split"])
        else:
            dataset = load_dataset(info["name"], split=info["split"])

        # Rename columns to match unified structure
        dataset = rename_column(info, dataset)

        datasets.append(dataset)

    # Concatenate all datasets
    combined_dataset = concatenate_datasets(datasets)

    # Save to disk
    combined_dataset.save_to_disk("combined_dataset")

    # Convert to JSONL
    convert_dataset_format(combined_dataset)

    print("✅ Combined dataset saved!")

# Run the script
combine_dataset()
