import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--sample_size', type=int, default=300_000, help="sample size for data")
    parser.add_argument('--data_name', type=str, default="eliplutchok/fineweb-small-sample", help="dataset name")
    parser.add_argument('--max_length', type=int, default=512, help="max sequence length")
    parser.add_argument('--stride', type=int, default=256, help="stride for sliding window")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="train/test split ratio")
    return parser.parse_args()

def preprocess_and_save(token_ids, max_length, stride, save_path):
    """ Optimized function that saves tokenized data in chunks """
    num_steps = (len(token_ids) - max_length) // stride + 1
    
    with open(save_path, "wb") as f:
        for i in tqdm(range(num_steps), desc=f"Processing {save_path}"):
            start_idx = i * stride
            input_chunk = token_ids[start_idx:start_idx + max_length]
            target_chunk = token_ids[start_idx + 1:start_idx + max_length + 1]
            np.save(f, np.array([input_chunk, target_chunk], dtype=np.int32))  # Save in compressed format

def split_and_save(token_ids, split_ratio=0.9, max_length=1024, stride=256):
    """ Split dataset and save efficiently """
    tokens_tensor = torch.tensor(token_ids)  # Convert only once
    num_samples = len(tokens_tensor)

    split_idx = int(num_samples * split_ratio)
    train_tokens, test_tokens = tokens_tensor[:split_idx], tokens_tensor[split_idx:]

    print(f"Total tokens: {num_samples:_}")
    print(f"Training on {len(train_tokens):_} tokens")
    print(f"Testing on {len(test_tokens):_} tokens")

    preprocess_and_save(train_tokens, max_length, stride, "train_data.npy")
    preprocess_and_save(test_tokens, max_length, stride, "test_data.npy")

def tokenize(tokenizer, data):
    """ Batch tokenize instead of one-by-one tokenization """
    texts = [item['text'] for item in data]
    encoded_data = tokenizer(texts, truncation=False, padding=False, return_tensors="np")["input_ids"]
    
    return np.concatenate(encoded_data)

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds = load_dataset(args.data_name, split='train')
    ds = ds.select(range(min(args.sample_size, len(ds))))
    tokens = tokenize(tokenizer, ds)  # Efficient tokenization
    
    split_and_save(tokens, args.split_ratio, args.max_length, args.stride)
