# Step 1: Install and Import Necessary Libraries
# ===============================================
# Ensure the required libraries are installed. This is for notebook environments.
!pip install -q "transformers[torch]>=4.28.0" datasets accelerate evaluate sentencepiece rouge_score sacrebleu huggingface_hub

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate

# Step 2: Verify GPU Availability
# =================================
# Check for GPU and print its name. Training will be much faster on a GPU.
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

# Step 3: Configure Model and Tokenizer
# =====================================
# Define the model checkpoint from Hugging Face Hub
MODEL_ID = "human-centered-summarization/financial-summarization-pegasus"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# Pegasus model does not have a pad_token, so we set it to the eos_token.
# This is crucial for correct padding.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 4: Load and Prepare the Dataset
# ====================================
# Load the financial news summarization dataset
DATASET_ID = "datht/FINDSum"
raw_dataset = load_dataset(DATASET_ID)

print("Dataset structure:")
print(raw_dataset)

# Determine the correct column names for the article and summary
input_col, target_col = "article", "summary"
print(f"\nUsing columns: Input='{input_col}', Target='{target_col}'")

# Define maximum sequence lengths for input and output
# Pegasus's max length is 1024
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128

# Define a function to filter out empty or None examples
def filter_empty_examples(example):
    return (
        example[input_col] is not None
        and len(example[input_col]) > 0
        and example[target_col] is not None
        and len(example[target_col]) > 0
    )

# Apply the filter to remove bad data
filtered_dataset = raw_dataset.filter(filter_empty_examples)
print("\nDataset structure after filtering empty examples:")
print(filtered_dataset)

# Step 5: Preprocessing Function
# ==============================
# This function tokenizes the input articles and target summaries.
def preprocess_function(examples):
    # Tokenize the input text, truncating to the max length
    model_inputs = tokenizer(
        examples[input_col],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    # Tokenize the target text (summaries) using the tokenizer as a target tokenizer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[target_col],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

    # The loss function in PyTorch ignores label IDs of -100.
    # We replace the pad_token_id in the labels with -100 to ensure
    # that padding is not considered when calculating the loss.
    label_ids = []
    for label_row in labels["input_ids"]:
        label_ids.append([(token if token != tokenizer.pad_token_id else -100) for token in label_row])
    
    model_inputs["labels"] = label_ids
    return model_inputs

# Apply the preprocessing function to all splits of the dataset
tokenized_dataset = filtered_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=filtered_dataset["train"].column_names, # Remove original text columns
)
print("\nTokenized dataset structure:")
print(tokenized_dataset)

# Step 6: Define Metrics for Evaluation
# =====================================
# Load the ROUGE metric for summarization evaluation
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Decode the generated summaries and reference summaries
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value * 100 for key, value in result.items()}

    # Add a metric for the length of the generated summaries
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Step 7: Configure and Run Training
# ==================================
# Define a data collator which will dynamically pad the batches
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="pegasus-financial-summary",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Reduce to 4 or 2 if you get memory errors
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True, # Essential for summarization metrics
    fp16=torch.cuda.is_available(), # Enable mixed-precision training if on GPU
    push_to_hub=False, # Set to True if you want to upload the model to HF Hub
)

# Create the Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start the training process
print("\nStarting model training...")
trainer.train()

# Evaluate the model on the test set after training is complete
print("\nEvaluating model on the test set...")
trainer.evaluate(eval_dataset=tokenized_dataset["test"])


# Step 8: Save the Fine-Tuned Model and Tokenizer
# ===============================================
# Save the model and tokenizer to the output directory
output_dir = training_args.output_dir
print(f"\nSaving model and tokenizer to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Step 9: Run Inference with the Fine-Tuned Model
# ===============================================
# Use the trained model for summarization
print("\nRunning inference with the fine-tuned model...")

# Example financial news article
input_text = """
NEW YORK (Reuters) - Global stock markets fell on Tuesday and the dollar strengthened
for a fourth straight day after a U.S. Federal Reserve official said the central bank
was still on track to raise interest rates this year, while oil prices slumped.
The U.S. central bank could still raise rates twice this year, with the first hike
possibly coming in September, Atlanta Fed President Dennis Lockhart said.
His comments followed data on Monday showing U.S. manufacturing activity slowed in July,
and a report on Friday that showed U.S. labor costs rose at a slower-than-expected
pace in the second quarter. The Dow Jones industrial average .DJI fell 86.08 points,
or 0.49 percent, to 17,512.4, the S&P 500 .SPX lost 9.53 points, or 0.45 percent,
to 2,093.79 and the Nasdaq Composite .IXIC dropped 26.17 points, or 0.51 percent,
to 5,102.41.
"""

# Tokenize the input text and generate the summary
inputs = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, return_tensors="pt", truncation=True)
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=150,
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n--- Example Summary ---")
print("Original Article:")
print(input_text)
print("\nGenerated Summary:")
print(summary)