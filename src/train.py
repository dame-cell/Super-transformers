import torch 
import gzip 
import wandb 
import argparse  
import numpy as np 
import torch.nn as nn 
from tqdm.auto import tqdm 
from model import build_model
from utils import GPTDatasetV1  
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
from generate import text_to_token_ids, token_ids_to_text, generate

# Set random seed for reproducibility
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--max_len', type=int, default=1024, help="context length for the model")
    parser.add_argument('--epoch', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--train_data', type=str, help="path to the train npz file")
    parser.add_argument('--test_data', type=str,  help="path to the test npz file")
    parser.add_argument('--wandb', type=bool, default=False, help="Use Weights and Biases for logging")
    parser.add_argument('--ssmax', type=bool, default=True, help="whether to use or not use scalable softmax")
    parser.add_argument('--use_pos_enc', type=bool, default=False, help="whether to use or not use Positional encodings")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size for training")
    parser.add_argument('--size', type=str, default="default", help="whether to use a small or default or large architecture ")
    parser.add_argument('--hf_data', type=str,default=None, help="Path to the Hugging Face dataset")
    parser.add_argument('--dataset_args', type=dict, help="Arguments for the Hugging Face dataset")
    parser.add_argument('--text_column', type=str, default="text", help="Text column in the dataset")
    parser.add_argument('--vocab_size', type=int, default=5000, help="Vocabulary size for the dataset")

    return parser.parse_args()

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)  # Faster transfer with non_blocking=True
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data  # If it's not a tensor, return as is


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="super-transformers", config=args)


    model = build_model(size=args.size, ssmax=args.ssmax, use_pos_enc=args.use_pos_enc)
    model.to(device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable CuDNN optimization

    train_dataset = GPTDatasetV1(args.train_data)
    val_dataset = GPTDatasetV1(args.test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, sampler=val_sampler, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epoch}", leave=True)

        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch, target_batch = to_device(input_batch, device), to_device(target_batch, device)

            optimizer.zero_grad()
            logits = model(input_batch)

            # Corrected loss computation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            progress_bar.update(1)
            

            if step % 100 == 0 and step > 0:
                step_loss = running_loss / (step + 1)  # Correct loss calculation
                print(f"Step {step+1} - Train Loss: {step_loss:.4f}")
    
                if args.wandb:
                    wandb.log({"step": step + 1, "train_loss": step_loss})  # Log step loss correctly


        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epoch} - Train Loss: {epoch_loss:.4f}")

        if step % 100 == 0 and step > 0:
              START_CONTEXT = "As an AI language model,"
              token_ids = generate(
                    model=model.module,
                    device=device,
                    idx=text_to_token_ids(START_CONTEXT, tokenizer),
                    max_new_tokens=20,
                    context_len=args.max_len,
                )
              sample_text = token_ids_to_text(token_ids, tokenizer)
              print(f"\nSample text:", sample_text)

        # Validation loop
        if step % 1000 == 0 and step > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_input, val_target in val_loader:
                    val_input, val_target = to_device(val_input, device), to_device(val_target, device)
                    val_logits = model(val_input)
                    val_loss += F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_target.view(-1)).item()

            epoch_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {epoch_val_loss:.4f}")

            if args.wandb:
                wandb.log({"step": step + 1, "val_loss": epoch_val_loss})

    print("Training Complete!")
    import os 
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "super_transformer.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    if args.wandb:
        wandb.finish()

  
if __name__ == "__main__":
  print("Training the model")
  args = parse_args()
  train(args) 

