import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset
import logging
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--sample_size', type=int, default=300_000, help="sample size for data")
    parser.add_argument('--data_name', type=str, default="eliplutchok/fineweb-small-sample", help="data to be used in processing and training")
    parser.add_argument('--max_length', type=int, default=512, help="maximum sequence length")
    parser.add_argument('--stride', type=int, default=256, help="stride for sliding window")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="train/test split ratio")
    parser.add_argument('--batch_size', type=int, default=1000, help="batch size for tokenization")
    parser.add_argument('--output_dir', type=str, default=".", help="directory to save processed data")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_args(args):
    if not (0 < args.split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1")
    if args.max_length <= 0 or args.stride <= 0:
        raise ValueError("max_length and stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

def preprocess_and_save(token_ids, max_length, stride, save_path):
    if len(token_ids) < max_length:
        logging.warning(f"Input sequence length {len(token_ids)} is less than max_length {max_length}")
        return
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    inputs, targets = [], []
    num_steps = (len(token_ids) - max_length) // stride + 1
    
    try:
        for i in tqdm(range(num_steps), desc=f"Processing {save_path.name}"):
            start_idx = i * stride
            input_chunk = token_ids[start_idx:start_idx + max_length]
            target_chunk = token_ids[start_idx + 1:start_idx + max_length + 1]
            
            # Ensure chunks are the correct length
            if len(input_chunk) == max_length and len(target_chunk) == max_length:
                inputs.append(input_chunk)
                targets.append(target_chunk)
        
        if inputs:  # Only save if we have valid sequences
            np.savez(save_path, inputs=np.array(inputs), targets=np.array(targets))
            logging.info(f"Saved {len(inputs)} sequences to {save_path}")
        else:
            logging.warning("No valid sequences were generated")
            
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

def tokenize(tokenizer, data, batch_size=1000):
    """Memory-efficient tokenization with batching"""
    all_tokens = []
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    try:
        for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing", total=total_batches):
            batch = data[i:i+batch_size]
            if isinstance(batch, dict):
                batch = batch["text"]
            
            # Add EOS token between sequences
            encoded_batch = tokenizer(
                batch,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=False,
                return_tensors=None
            )["input_ids"]
            
            # Add EOS token between sequences
            for seq in encoded_batch:
                all_tokens.extend(seq + [tokenizer.eos_token_id])
        
        return all_tokens
    except Exception as e:
        logging.error(f"Error during tokenization: {str(e)}")
        raise

def split_and_save(token_ids, args):
    try:
        tokens_tensor = torch.tensor(token_ids)
        num_samples = len(tokens_tensor)
        
        split_idx = int(num_samples * args.split_ratio)
        train_tokens = tokens_tensor[:split_idx].numpy()
        test_tokens = tokens_tensor[split_idx:].numpy()
        
        logging.info(f"Total tokens: {num_samples:,}")
        logging.info(f"Training tokens: {len(train_tokens):,}")
        logging.info(f"Testing tokens: {len(test_tokens):,}")
        
        train_sequences = (len(train_tokens) - args.max_length) // args.stride + 1
        test_sequences = (len(test_tokens) - args.max_length) // args.stride + 1
        logging.info(f"Expected training sequences: {train_sequences:,}")
        logging.info(f"Expected testing sequences: {test_sequences:,}")

        output_dir = Path(args.output_dir)
        preprocess_and_save(train_tokens, args.max_length, args.stride, output_dir / "train_data.npz")
        preprocess_and_save(test_tokens, args.max_length, args.stride, output_dir / "test_data.npz")
        
    except Exception as e:
        logging.error(f"Error during split and save: {str(e)}")
        raise

def main():
    args = parse_args()
    setup_logging()
    validate_args(args)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=True)
        ds = load_dataset(args.data_name, split='train')
        
        if args.sample_size > len(ds):
            logging.warning(f"Requested sample size {args.sample_size} is larger than dataset size {len(ds)}")
            args.sample_size = len(ds)
        
        ds = ds.select(range(args.sample_size))
        tokens = tokenize(tokenizer, ds, args.batch_size)
        split_and_save(tokens, args)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()