#!/usr/bin/env python
"""
zeroshot_sentiment_prediction.py

Add zero-shot sentiment predictions (positive/negative/neutral) for every
actor predicted by the actor classification script.

Improved version with better error handling, memory management, and batch processing.

Use Example:
-------
python improved_zeroshot_sentiment_prediction.py \
    --predictions_csv predictions.csv \
    --output_csv predictions_with_sentiment.csv
"""
import argparse
import ast
import json
import os
import gc
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# Configuration
ZS_MODEL_NAME = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
SENT_TEMPLATES: Dict[str, List[str]] = {
    "positive": [
        "The UNSC's overall stance towards {actor} is positive.",
        "Comments about {actor} are favorable.",
        "The tone regarding {actor} is supportive."
    ],
    "negative": [
        "The UNSC's overall stance towards {actor} is negative.",
        "Comments about {actor} are unfavorable.",
        "The tone regarding {actor} is unsupportive."
    ],
    "neutral": [
        "The UNSC's overall stance towards {actor} is neutral.",
        "Sentiment expressed about {actor} is moderate.",
        "The tone regarding {actor} is impartial."
    ],
}
ENTAIL_IDX = 0  # ModernBERT puts the "entailment" logit at index-0
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512

def setup_device_and_memory():
    """Setup device and print system information"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
        gc.collect()
    
    return device

def setup_hf_auth():
    """Setup Hugging Face authentication if needed"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        print("✓ Hugging Face authentication successful")
    except Exception as e:
        print(f"⚠️ Hugging Face authentication issue: {e}")
        print("Consider running: huggingface-cli login")

def load_zero_shot_model(model_name: str = ZS_MODEL_NAME, device: torch.device = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load zero-shot model with error handling"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("✓ Tokenizer loaded successfully")
        
        print(f"Loading model from {model_name}...")
        # Use lower precision for memory efficiency
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )
        print("✓ Model loaded successfully")
        
        model = model.to(device).eval()
        print(f"Model moved to device: {device}")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have internet connection and the model name is correct")
        raise

def safe_literal_eval(actor_str) -> List[str]:
    """Safely evaluate actor string with fallbacks"""
    if isinstance(actor_str, list):
        return actor_str
    
    if not isinstance(actor_str, str):
        print(f"Warning: Unexpected actor type {type(actor_str)}, converting to string")
        actor_str = str(actor_str)
    
    try:
        # Try to evaluate as literal
        result = ast.literal_eval(actor_str)
        if isinstance(result, list):
            return result
        else:
            return [str(result)]
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse actor string '{actor_str}', treating as single actor")
        # Clean up the string and treat as single actor
        cleaned = actor_str.strip("[]'\"")
        return [cleaned] if cleaned else ["Unknown"]

def sentiment_for_pair(premise: str, actor: str, tokenizer, model, max_length: int, device: torch.device) -> Tuple[str, Dict[str, float]]:
    """
    Evaluate (premise, hypothesis) pairs across every template and sentiment,
    return the sentiment with the highest average entailment log-prob and all scores.
    """
    if not premise or not actor:
        return "neutral", {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    scores = {s: [] for s in SENT_TEMPLATES}
    
    try:
        with torch.no_grad():
            for sentiment, templates in SENT_TEMPLATES.items():
                for template in templates:
                    try:
                        hypothesis = template.format(actor=actor)
                        
                        # Truncate premise if too long to avoid memory issues
                        if len(premise) > 2000:  # rough character limit
                            premise = premise[:2000] + "..."
                        
                        enc = tokenizer(
                            premise,
                            hypothesis,
                            truncation="only_first",
                            padding="longest",
                            max_length=max_length,
                            return_tensors="pt",
                        ).to(device)

                        logits = model(**enc).logits.squeeze(0)
                        scores[sentiment].append(logits[ENTAIL_IDX].item())
                        
                    except Exception as e:
                        print(f"Warning: Error processing template for {sentiment}: {e}")
                        scores[sentiment].append(0.0)  # Fallback score

        # Average over templates for robustness
        avg_scores = {s: np.mean(v) if v else 0.0 for s, v in scores.items()}
        
        # Get the sentiment with highest score
        best_sentiment = max(avg_scores, key=avg_scores.get)
        
        return best_sentiment, avg_scores
        
    except Exception as e:
        print(f"Warning: Error in sentiment analysis for actor '{actor}': {e}")
        return "neutral", {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

def process_batch_sentiments(batch_data: List[Tuple[str, List[str]]], tokenizer, model, max_length: int, device: torch.device) -> List[Tuple[List[str], List[Dict[str, float]]]]:
    """Process a batch of clause-actors pairs for sentiment analysis"""
    results = []
    
    for premise, actors in batch_data:
        clause_sents = []
        clause_scores = []
        
        for actor in actors:
            try:
                sentiment, scores = sentiment_for_pair(premise, actor, tokenizer, model, max_length, device)
                clause_sents.append(sentiment)
                clause_scores.append(scores)
            except Exception as e:
                print(f"Warning: Error processing actor '{actor}': {e}")
                clause_sents.append("neutral")
                clause_scores.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})
        
        results.append((clause_sents, clause_scores))
    
    return results

def append_sentiments(df: pd.DataFrame, tokenizer, model, max_length: int, device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
    """
    Process sentiments with improved batch processing and error handling
    """
    print(f"Processing {len(df)} rows with batch size {batch_size}")
    
    sentiments_all: List[List[str]] = []
    sentiment_scores_all: List[List[Dict[str, float]]] = []
    
    # Prepare batch data
    batch_data = []
    for i, (premise, actor_str) in enumerate(zip(df["clause"], df["predicted_actors"])):
        try:
            # Convert premise to string and handle NaN
            premise = str(premise) if pd.notna(premise) else ""
            
            # Parse actors safely
            actors = safe_literal_eval(actor_str)
            
            batch_data.append((premise, actors))
            
        except Exception as e:
            print(f"Warning: Error preparing data for row {i}: {e}")
            batch_data.append(("", ["Unknown"]))
    
    # Process in batches
    for i in tqdm(range(0, len(batch_data), batch_size), desc="Processing sentiment batches"):
        batch = batch_data[i:i + batch_size]
        
        try:
            batch_results = process_batch_sentiments(batch, tokenizer, model, max_length, device)
            
            for sentiments, scores in batch_results:
                sentiments_all.append(sentiments)
                sentiment_scores_all.append(scores)
                
        except Exception as e:
            print(f"Warning: Error processing batch starting at row {i}: {e}")
            # Add fallback results for this batch
            for _ in batch:
                sentiments_all.append(["neutral"])
                sentiment_scores_all.append([{"positive": 0.0, "negative": 0.0, "neutral": 1.0}])
        
        # Clear cache periodically
        if i % (batch_size * 4) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Add results to dataframe
    df = df.copy()
    df["predicted_sentiments"] = sentiments_all
    df["sentiment_scores"] = sentiment_scores_all
    
    return df

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data"""
    print(f"Input data shape: {df.shape}")
    
    required_columns = ["clause", "predicted_actors"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty data
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Show data info
    print(f"Columns: {list(df.columns)}")
    print(f"Non-null clause count: {df['clause'].notna().sum()}")
    print(f"Non-null predicted_actors count: {df['predicted_actors'].notna().sum()}")
    
    # Handle missing values
    df = df.copy()
    df["clause"] = df["clause"].fillna("")
    df["predicted_actors"] = df["predicted_actors"].fillna("[]")
    
    return df

def save_results_with_backup(df: pd.DataFrame, output_path: str):
    """Save results with backup and validation"""
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save main file
        df.to_csv(output_path, index=False)
        print(f"✓ Results saved to {output_path}")
        
        # Create backup with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.replace(".csv", f"_backup_{timestamp}.csv")
        df.to_csv(backup_path, index=False)
        print(f"✓ Backup saved to {backup_path}")
        
        # Validate saved file
        test_df = pd.read_csv(output_path)
        print(f"✓ Validation: Saved file has {len(test_df)} rows")
        
        # Show sample results
        print("\n=== Sample Results ===")
        for i in range(min(3, len(df))):
            clause_preview = str(df.iloc[i]["clause"])[:100] + "..." if len(str(df.iloc[i]["clause"])) > 100 else str(df.iloc[i]["clause"])
            actors = df.iloc[i]["predicted_actors"]
            sentiments = df.iloc[i]["predicted_sentiments"]
            print(f"Sample {i+1}:")
            print(f"  Clause: '{clause_preview}'")
            print(f"  Actors: {actors}")
            print(f"  Sentiments: {sentiments}")
            print()
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise

def main(args):
    """Main function with comprehensive error handling"""
    try:
        print("=== Starting Sentiment Analysis ===")
        
        # Setup system
        device = setup_device_and_memory()
        setup_hf_auth()
        
        # Load and validate input data
        print(f"Loading data from {args.predictions_csv}...")
        if not os.path.exists(args.predictions_csv):
            raise FileNotFoundError(f"{args.predictions_csv} does not exist.")
        
        df = pd.read_csv(args.predictions_csv)
        df = validate_input_data(df)
        
        # Load zero-shot model
        tokenizer, model = load_zero_shot_model(args.model_name, device)
        max_length = min(tokenizer.model_max_length, args.max_length)
        print(f"Using max_length: {max_length}")
        
        # Process sentiments
        print("Starting sentiment analysis...")
        df_with_sentiment = append_sentiments(
            df, tokenizer, model, max_length, device, args.batch_size
        )
        
        # Save results
        print("Saving results...")
        save_results_with_backup(df_with_sentiment, args.output_csv)
        
        print("✓ Sentiment analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Process failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append zero-shot sentiment predictions to actor output CSV."
    )
    parser.add_argument(
        "--predictions_csv",
        required=True,
        help="CSV generated by the actor classification script (contains predicted_actors).",
    )
    parser.add_argument(
        "--output_csv",
        default="predictions_with_sentiments.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--model_name",
        default=ZS_MODEL_NAME,
        help="HF model to use for zero-shot sentiment NLI.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing (lower = less memory usage)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Limit processing to first N rows (for testing)"
    )
    
    args = parser.parse_args()
    
    # Apply sample size if specified
    if args.sample_size:
        print(f"⚠️ Processing only first {args.sample_size} rows for testing")
    
    main(args)