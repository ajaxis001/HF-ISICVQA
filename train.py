from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np

import evaluate 

dataset = load_datasest("yelp_review_full") # Load dataset
print(f"Example from dataset : {dataset['train'][10]}") # Print an example from the dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Load tokenizer

def tokenize_function(examples):
    """ Tokenize the text column in the dataset"""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datases = dataset.map(tokenize_function, batched=True) # Tokenize the dataset
small_train_dataset = tokenized_datasets["train"].shuffle(seed=44).select(range(1000)) # Select a small subset of the dataset for training
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=44).select(range(1000)) # Select a small subset of the dataset for evaluation

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5) # Load model

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch") # Create training arguments
metric = evaluate.load("accuracy") # Load metric

def compute_metrics(eval_pred):
    """ Compute metrics from the evaluation dataset """
    logits, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train() # Train the models