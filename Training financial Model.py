import torch
print("CUDA available?", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")




!pip install -q "transformers[torch]" datasets accelerate evaluate sentencepiece rouge_score sacrebleu
!pip install -q huggingface_hub



import os
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
print("torch.cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")




MODEL_ID = "human-centered-summarization/financial-summarization-pegasus"  # HF checkpoint
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)




DATASET_ID = "datht/FINDSum"   # public FINDSum dataset on HF
raw = load_dataset(DATASET_ID)
print(raw)          # shows train/validation/test and column names
# inspect column names quickly:
print("train columns:", raw["train"].column_names)





# candidate names (covers common variants)
input_candidates  = ["article", "document", "text", "body", "content"]
target_candidates = ["summary", "highlights", "abstract", "summary_text"]
def find_columns(dataset):
    cols = dataset.column_names
    inp = next((c for c in input_candidates if c in cols), None)
    tgt = next((c for c in target_candidates if c in cols), None)
    if inp is None or tgt is None:
        raise ValueError(f"Could not find input/target columns. Available: {cols}")
    return inp, tgt
input_col, target_col = find_columns(raw["train"])
print("Using:", input_col, "->", target_col)



# set lengths; use tokenizer.model_max_length as a safe ceiling
# depends on: PVZuEw3XpDMU
max_input_length  = min(1024, tokenizer.model_max_length or 512)
max_target_length = 128   # adjust if you want longer summaries
def preprocess_function(examples):
    inputs = examples[input_col]
    targets = examples[target_col]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    # replace pad token id's in labels by -100 for PyTorch loss ignore
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs
# Suggestion: Add this after loading the dataset
def filter_empty(example):
    return example[input_col] is not None and len(example[input_col]) > 0 and \
           example[target_col] is not None and len(example[target_col]) > 0

raw = raw.filter(filter_empty)
# apply mapping (use num_proc if Colab supports multiprocessing)
tokenized = raw.map(preprocess_function, batched=True, remove_columns=raw["train"].column_names)
print(tokenized)



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
rouge = evaluate.load("rouge")
def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace -100 with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # round the results
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len"] = float(np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]))
    return result



from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Set fp16 only if a GPU is present
)



# Corrected code ✅
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"], # Use the "validation" split for evaluation
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()

trainer.evaluate()


#model.save_pretrained("./results")
#tokenizer.save_pretrained("./results")


input_text = "Your financial news article text here."
inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)


