#!/usr/bin/env python
"""
actor_classification.py

Multi-label classify the actors in some number of clauses.
Improved version with better error handling and Colab compatibility.
"""

import argparse
import os
import json
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import gc
from typing import Optional

from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
from datetime import datetime

# Check for Hugging Face token
def setup_hf_auth():
    """Setup Hugging Face authentication if needed"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Test if we can access HF
        api.whoami()
        print("✓ Hugging Face authentication successful")
    except Exception as e:
        print(f"⚠️ Hugging Face authentication issue: {e}")
        print("Consider running: huggingface-cli login")
        print("Or use: from huggingface_hub import login; login()")

# custom binary cross entropy loss trainer
class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss

# metrics being tracked
accuracy_metric  = evaluate.load("accuracy")
f1_metric        = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")

def multi_label_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits))
    preds = (probs >= 0.5).int().numpy()
    flat_preds = preds.reshape(-1)
    flat_labels = labels.reshape(-1)
    return {
        "accuracy":  accuracy_metric.compute(predictions=flat_preds, references=flat_labels)["accuracy"],
        "f1":        f1_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["f1"],
        "precision": precision_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["precision"],
        "recall":    recall_metric.compute(predictions=flat_preds, references=flat_labels, average="macro")["recall"],
    }

# sliding window tokenizer
def chunk_and_tokenize(examples, tokenizer, max_length=512, stride=256):
    all_input_ids, all_attention_masks, all_labels = [], [], []
    for clause, label_vec in zip(examples["clause"], examples["labels"]):
        tokens = tokenizer.encode(clause, add_special_tokens=True)
        for start in range(0, len(tokens), stride):
            window = tokens[start:start+max_length]
            mask = [1]*len(window)
            pad_len = max_length - len(window)
            if pad_len > 0:
                window += [tokenizer.pad_token_id]*pad_len
                mask += [0]*pad_len
            all_input_ids.append(window)
            all_attention_masks.append(mask)
            all_labels.append(label_vec)
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }

# global variable declarations
BASE_MODEL_NAME = "rwillh11/mdeberta_groups_2.0"
MAX_LENGTH = 512
STRIDE = 256

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def check_system_info():
    """Print system information for debugging"""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("⚠️ Transformers not installed")

def train(data_file, output_dir):
    """Training function with improved error handling"""
    try:
        print("=== Starting Training ===")
        check_system_info()
        clear_gpu_cache()
        
        # load and parse the raw lists (e.g. ['Government', 'Rebels', ...])
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples")
        
        df["actors"] = df["regularized_actors"].apply(ast.literal_eval)

        # binarize the labels
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df["actors"])
        df_labels = pd.DataFrame(y, columns=mlb.classes_)
        df = pd.concat([df, df_labels], axis=1)
        df["labels"] = df[mlb.classes_].values.tolist()

        # explicitly set label2id/id2label for new head
        label_list = mlb.classes_.tolist()
        id2label = {i: lbl for i, lbl in enumerate(label_list)}
        label2id = {lbl: i for i, lbl in enumerate(label_list)}
        print(f"Number of labels: {len(label_list)}")

        # build HF dataset and split
        ds = Dataset.from_pandas(df[["clause","labels"]])
        split = ds.train_test_split(test_size=0.2, seed=42)
        train_ds, val_ds = split["train"], split["test"]

        # tokenize and chunk each example into overlapping windows
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
        
        print("Tokenizing datasets...")
        train_tokenized = train_ds.map(
            lambda ex: chunk_and_tokenize(ex, tokenizer, MAX_LENGTH, STRIDE),
            batched=True,
            remove_columns=train_ds.column_names,
        )
        val_tokenized = val_ds.map(
            lambda ex: chunk_and_tokenize(ex, tokenizer, MAX_LENGTH, STRIDE),
            batched=True,
            remove_columns=val_ds.column_names,
        )

        # format for pytorch training
        train_tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        val_tokenized.set_format(type="torch",   columns=["input_ids","attention_mask","labels"])

        # model and config setup (with the explicit id2label/label2id)
        print("Loading model configuration...")
        config = AutoConfig.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        
        # load the new model
        print("Loading model...")
        new_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True
        )

        # show which labels exist in the new head
        new_lab2id = new_model.config.label2id
        print("New head labels:", list(new_lab2id.keys()))

        # warm start the head from the original 44-way model
        print("Performing warm start...")
        old_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            problem_type="multi_label_classification",
        )
        old_lab2id = old_model.config.label2id

        # define the warm-start map
        warm_map = {
            "Donors":                       "Investors And Stakeholders",
            "Government":                   "Politicians",
            "Mediators":                    "Civil Servants",
            "Member States":                "Citizens",
            "Pirates":                      "Criminals",
            "Rebels":                       "Criminals",
            "Refugees":                     "Migrants And Refugees",
            "Regional Organizations":       "Ethnic And National Communities",
            "Regional Stakeholders":        "Ethnic And National Communities",
            "Peacekeeping":                 "Military Personnel",
            "Troop Contributing Countries": "Military Personnel",
            "Terrorists":                   "Criminals",
            "Stakeholders":                 "Investors And Stakeholders",
            "Secretary General":            "Politicians",
            "Security Council":             "Politicians",
            "Permanent Members":            "Politicians",
        }

        # filter for valid mappings
        valid_warm = {
            new_lab: old_lab
            for new_lab, old_lab in warm_map.items()
            if new_lab in new_lab2id and old_lab in old_lab2id
        }
        skipped = set(warm_map) - set(valid_warm)
        if skipped:
            print(f"WARNING--Skipping warm-start for missing labels: {skipped}")

        # copy weights and biases for valid labels
        old_w = old_model.classifier.weight.data
        old_b = old_model.classifier.bias.data
        new_w = new_model.classifier.weight.data
        new_b = new_model.classifier.bias.data
        for new_lab, old_lab in valid_warm.items():
            ni = new_lab2id[new_lab]
            oi = old_lab2id[old_lab]
            new_w[ni].copy_(old_w[oi])
            new_b[ni].copy_(old_b[oi])

        model = new_model
        del old_model  # Free memory
        clear_gpu_cache()

        os.makedirs(output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=500,  # Evaluate every 500 steps
            save_strategy="steps",
            save_steps=500,  # Save every 500 steps
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=2e-5,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            seed=42,
            report_to="none",
            dataloader_pin_memory=False,  # Reduce memory usage
        )

        data_collator = DataCollatorWithPadding(tokenizer)
        trainer = BCETrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=multi_label_metrics
        )

        # train and evaluate
        print("Starting training...")
        trainer.train()
        eval_results = trainer.evaluate()
        print("Final evaluation:", eval_results)

        trainer.save_model(output_dir)
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

        # save the label classes for inference
        with open(os.path.join(output_dir, "label_map.json"), "w") as f:
            json.dump({"classes": mlb.classes_.tolist()}, f, indent=2)
            
        print(f"✓ Training completed successfully! Model saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

def predict(data_file, model_dir, output_file, sample_size: Optional[int] = None, batch_size: int = 16):
    """
    Prediction function with improved error handling and memory management
    
    Args:
        data_file: Path to CSV file with data to predict
        model_dir: Hugging Face model name or local model directory
        output_file: Path to save predictions
        sample_size: Optional limit on number of samples to process
        batch_size: Batch size for processing (smaller = less memory)
    """
    try:
        print("=== Starting Prediction ===")
        check_system_info()
        setup_hf_auth()
        clear_gpu_cache()
        
        print(f"Loading tokenizer from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        print("✓ Tokenizer loaded successfully")
        
        print(f"Loading model from {model_dir}...")
        # Use lower precision for memory efficiency
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype
        )
        print("✓ Model loaded successfully")
        print(f"Model config: {model.config}")

        # Derive class list from model config
        id2label = model.config.id2label
        classes = [lbl for _, lbl in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
        print(f"Number of classes: {len(classes)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        print(f"Model moved to device: {device}")

        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples")

        # Apply sampling if specified
        if sample_size and sample_size < len(df):
            print(f"Sampling {sample_size} observations from {len(df)} total")
            df = df.sample(n=sample_size, random_state=42)
        else:
            print(f"Processing all {len(df)} observations")
        
        df["clause"] = df["clause"].astype(str)

        print("Starting predictions...")
        preds_list = []
        total_samples = len(df)
        
        # Process in batches to manage memory
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_clauses = df["clause"].iloc[i:batch_end].tolist()
            
            print(f"Processing batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1} "
                  f"({i+1}-{batch_end}/{total_samples})")
            
            batch_preds = []
            for clause in batch_clauses:
                try:
                    enc = tokenizer(
                        clause,
                        return_overflowing_tokens=True,
                        truncation=True,
                        padding="max_length",       
                        max_length=MAX_LENGTH,
                        stride=STRIDE,
                        return_tensors="pt"
                    )
                    input_ids = enc["input_ids"].to(device)
                    mask = enc["attention_mask"].to(device)
                    
                    with torch.no_grad():
                        logits = model(input_ids=input_ids, attention_mask=mask).logits
                        probs = torch.sigmoid(logits).mean(dim=0).cpu().numpy()
                    
                    # Predict labels with threshold
                    preds = [classes[i] for i, p in enumerate(probs) if p >= 0.5]
                    if not preds:  # Ensure at least one prediction
                        preds = [classes[int(np.argmax(probs))]]
                    
                    batch_preds.append(preds)
                    
                except Exception as e:
                    print(f"Warning: Error processing clause, skipping: {e}")
                    batch_preds.append(["Unknown"])  # Fallback
            
            preds_list.extend(batch_preds)
            
            # Clear cache periodically
            if i % (batch_size * 4) == 0:
                clear_gpu_cache()

        print("Adding predictions to dataframe...")
        df["predicted_actors"] = preds_list
        
        print(f"Saving predictions to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"✓ Predictions saved successfully to {output_file}")
        print(f"Processed {len(df)} samples")
        
        # Show sample predictions
        print("\nSample predictions:")
        for i in range(min(3, len(df))):
            clause_preview = df.iloc[i]["clause"][:100] + "..." if len(df.iloc[i]["clause"]) > 100 else df.iloc[i]["clause"]
            print(f"  Sample {i+1}: '{clause_preview}' -> {df.iloc[i]['predicted_actors']}")
        
    except Exception as e:
        print(f"❌ Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Actor Classification Tool")
    p.add_argument("--mode", choices=["train","predict"], required=True,
                   help="Mode: train a new model or predict with existing model")
    p.add_argument("--data_file", type=str, required=True,
                   help="Path to CSV file with data")
    p.add_argument("--output_dir", type=str, default="./model_out",
                   help="Directory to save trained model")
    p.add_argument("--model_dir", type=str,
                   help="Hugging Face model name or local directory for prediction")
    p.add_argument("--output_file", type=str, default="predictions.csv",
                   help="Output CSV file for predictions")
    p.add_argument("--sample_size", type=int,
                   help="Limit number of samples to process (optional)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for prediction (lower = less memory)")
    
    args = p.parse_args()

    if args.mode == "train":
        train(args.data_file, args.output_dir)
    else:
        if not args.model_dir:
            raise ValueError("--model_dir is required for predict mode")
        predict(args.data_file, args.model_dir, args.output_file, 
                args.sample_size, args.batch_size)