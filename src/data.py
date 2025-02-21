import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--sample_size', type=int, default=300_000, help="sample size for data")
    parser.add_argument('--data_name', type=str, default="eliplutchok/fineweb-small-sample", help="data to be used in processing and training")
    parser.add_argument('--max_length', type=int, default=512, help="maximum sequence length")
    parser.add_argument('--stride', type=int, default=256, help="stride for sliding window")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="train/test split ratio")
    return parser.parse_args()

def preprocess_and_save(token_ids, max_length, stride, save_path):
    inputs, targets = [], []
    
    num_steps = (len(token_ids) - max_length) // stride + 1
    for i in tqdm(range(num_steps), desc=f"Processing {save_path}"):
        start_idx = i * stride
        input_chunk = token_ids[start_idx:start_idx + max_length]
        target_chunk = token_ids[start_idx + 1:start_idx + max_length + 1]
        inputs.append(input_chunk)
        targets.append(target_chunk)
    
    np.savez(save_path, inputs=np.array(inputs), targets=np.array(targets))


# Maybe find a better way to tokenize and batch for better Space complexity 
def tokenize(tokenizer, data, batch_size=1000):
    encoded_data = []
    
    for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing"):
        batch = data[i:i+batch_size] 
        if isinstance(batch, dict):  
            batch = batch["text"] 
        
        encoded_batch = tokenizer(batch, truncation=False, padding=False)["input_ids"]
        encoded_data.extend(encoded_batch)
    
    return [token for text in encoded_data for token in text]


def split_and_save(token_ids, split_ratio=0.9, max_length=1024, stride=256):
    tokens_tensor = torch.tensor(token_ids)
    num_samples = len(tokens_tensor)
    
    split_idx = int(num_samples * split_ratio)
    train_tokens, test_tokens = tokens_tensor[:split_idx], tokens_tensor[split_idx:]
    
    print(f"Total tokens: {num_samples:_}")
    print(f"Training on {len(train_tokens):_} tokens")
    print(f"Testing on {len(test_tokens):_} tokens")
    
    print(f"Number of training sequences: {(len(train_tokens) - max_length) // stride + 1:_}")
    print(f"Number of testing sequences: {(len(test_tokens) - max_length) // stride + 1:_}")

    preprocess_and_save(train_tokens, max_length, stride, "train_data.npz")
    preprocess_and_save(test_tokens, max_length, stride, "test_data.npz")



if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=True)

    ds = load_dataset(args.data_name, split='train')
    ds = ds.select(range(min(args.sample_size, len(ds))))
    tokens = tokenize(tokenizer, ds)

    split_and_save(tokens, args.split_ratio, args.max_length, args.stride)
