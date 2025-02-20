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
    inputs = []
    targets = []
    
    # Calculate the number of steps for tqdm
    num_steps = (len(token_ids) - max_length) // stride + 1
    
    # Preprocess token_ids with tqdm to show progress
    for i in tqdm(range(num_steps), desc=f"Processing {save_path}"):
        start_idx = i * stride
        input_chunk = token_ids[start_idx:start_idx + max_length]
        target_chunk = token_ids[start_idx + 1:start_idx + max_length + 1]
        inputs.append(input_chunk)
        targets.append(target_chunk)
    
    np.savez(save_path, inputs=np.array(inputs), targets=np.array(targets))

def split_and_save(token_ids, split_ratio=0.9, max_length=1024, stride=256):
    tokens_tensor = torch.tensor(token_ids)
    num_samples = len(tokens_tensor)
    
    split_idx = int(num_samples * split_ratio)
    train_tokens = tokens_tensor[:split_idx]
    test_tokens = tokens_tensor[split_idx:]
    
    total_train_tokens = len(train_tokens)
    total_test_tokens = len(test_tokens)
    
    # Printing the number of tokens in a readable format
    print(f"Total tokens: {num_samples:_}")
    print(f"Training on {total_train_tokens:_} tokens")
    print(f"Testing on {total_test_tokens:_} tokens")
    
    # Calculate and print the number of sequences
    train_sequences = (len(train_tokens) - max_length) // stride + 1
    test_sequences = (len(test_tokens) - max_length) // stride + 1
    print(f"Number of training sequences: {train_sequences:_}")
    print(f"Number of testing sequences: {test_sequences:_}")
    
    # Preprocess and save train and test data
    preprocess_and_save(train_tokens, max_length, stride, "train_data.npz")
    preprocess_and_save(test_tokens, max_length, stride, "test_data.npz")


def tokenize(tokenizer,data):
    encoded_data = []
    
    # Use tqdm for progress bar
    for item in tqdm(data, desc="Tokenizing"):
        encoded_text = tokenizer.encode(item['text'])
        encoded_data.append(encoded_text)
    
    return [token for text in encoded_data for token in text]

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    ds = load_dataset(args.data_name, split='train')
    ds = ds.select(range(min(args.sample_size, len(ds))))
    tokens = tokenize(tokenizer, ds)
    
    split_and_save(tokens, args.split_ratio, args.max_length, args.stride)