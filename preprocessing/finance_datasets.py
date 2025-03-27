from datasets import load_dataset, concatenate_datasets


# load dataset
dataset_1 = load_dataset("gbharti/finance-alpaca")
dataset_2 = load_dataset("PaulAdversarial/all_news_finance_sm_1h2023")
dataset_3 = load_dataset("winddude/reddit_finance_43_250k")
dataset_4 = load_dataset("causal-lm/finance")

# create a column called text
dataset_1 = dataset_1.map(
    lambda example: {"text": example["instruction"] + " " + example["output"]},
    num_proc=4,
)
dataset_1 = dataset_1.remove_columns(["input", "instruction", "output"])

dataset_2 = dataset_2.map(
    lambda example: {"text": example["title"] + " " + example["description"]},
    num_proc=4,
)
dataset_2 = dataset_2.remove_columns(
    ["_id", "main_domain", "title", "description", "created_at"]
)

dataset_3 = dataset_3.map(
    lambda example: {
        "text": example["title"] + " " + example["selftext"] + " " + example["body"]
    },
    num_proc=4,
)
dataset_3 = dataset_3.remove_columns(
    [
        "id",
        "title",
        "selftext",
        "z_score",
        "normalized_score",
        "subreddit",
        "body",
        "comment_normalized_score",
        "combined_score",
    ]
)

dataset_4 = dataset_4.map(
    lambda example: {"text": example["instruction"] + " " + example["output"]},
    num_proc=4,
)
dataset_4 = dataset_4.remove_columns(["input", "instruction", "output"])

# combine and split train test sets
combined_dataset = concatenate_datasets(
    [
        dataset_1["train"],
        dataset_2["train"],
        dataset_3["train"],
        dataset_4["train"],
        dataset_4["validation"],
    ]
)

datasets = combined_dataset.train_test_split(test_size=0.2)
