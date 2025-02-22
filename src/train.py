import os 
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
from transformers import get_linear_schedule_with_warmup
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
    parser.add_argument('--vocab_size', type=int, default=50257, help="Vocabulary size for the dataset")
    parser.add_argument('--generating_step', type=int, default=2000, help="what step to generate during training")
    parser.add_argument('--validation_step', type=int, default=1000, help="what step to validate  during training")
    parser.add_argument('--save_model', type=int, default=1000, help="what step to save  during training")


    return parser.parse_args()

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)  # Faster transfer with non_blocking=True
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data  # If it's not a tensor, return as is




def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    if args.wandb:
        wandb.login(key="04098c64a0b88d5f4ff90335b7f75613041420c6")
        wandb.init(project="super-transformers", config=args)

    # Model setup
    model = build_model(size=args.size, ssmax=args.ssmax, use_pos_enc=args.use_pos_enc)
    model.to(device)
    
    # Dataset and loader setup
    train_dataset = GPTDatasetV1(args.train_data)
    val_dataset = GPTDatasetV1(args.test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),  # Modified beta2 for better stability
        eps=1e-8,
        weight_decay=0.1    # L2 regularization
    )

    # Learning rate schedule setup
    num_training_steps = len(train_loader) * args.epoch
    num_warmup_steps = min(1000, int(0.05 * num_training_steps))  # Shorter warmup for single epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Training state tracking
    best_val_loss = float("inf")
    best_model_path = "saved_models/best_model.pth"
    os.makedirs("saved_models", exist_ok=True)
    
    # Gradient accumulation setup
    accumulation_steps = 4  # Adjust based on your batch size
    
    def evaluate_model():
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_input, val_target in tqdm(val_loader, desc="Validating", leave=False):
                val_input, val_target = to_device(val_input, device), to_device(val_target, device)
                val_logits = model(val_input)
                val_loss += F.cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)), 
                    val_target.view(-1),
                    label_smoothing=0.1  # Add label smoothing
                ).item()
        return val_loss / len(val_loader)

    def generate_sample():
        START_CONTEXT = "As an AI language model,"
        token_ids = generate(
            model=model,
            device=device,
            idx=text_to_token_ids(START_CONTEXT, tokenizer),
            max_new_tokens=20,
            context_len=args.max_len,
            temperature=0.8  # Add temperature for better sampling
        )
        return token_ids_to_text(token_ids, tokenizer)

    # Training loop
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epoch}", leave=True)
        
        # Learning rate warmup tracking
        if epoch == 0:
            print(f"Warmup steps: {num_warmup_steps}")
            print(f"Total steps: {num_training_steps}")

        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch, target_batch = to_device(input_batch, device), to_device(target_batch, device)
            
            # Forward pass
            logits = model(input_batch)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target_batch.view(-1),
                label_smoothing=0.1  # Add label smoothing
            )
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Gradient tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update on accumulation boundary
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Loss tracking
            running_loss += loss.item() * accumulation_steps
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}",
                'grad': f"{grad_norm:.2f}"
            })
            progress_bar.update(1)

            # Periodic logging
            if step % 500 == 0 and step > 0:
                avg_loss = running_loss / (step + 1)
                print(f"\nStep {step+1} - Train Loss: {avg_loss:.4f} - LR: {current_lr:.2e} - Grad Norm: {grad_norm:.2f}")
                if args.wandb:
                    wandb.log({
                        "step": step + 1,
                        "train_loss": avg_loss,
                        "learning_rate": current_lr,
                        "gradient_norm": grad_norm
                    })

            # Generate sample text
            if step % args.generating_step == 0 and step > 0:
                sample_text = generate_sample()
                print(f"\nSample text: {sample_text}")

            # Validation
            if step % args.validation_step == 0 and step > 0:
                epoch_val_loss = evaluate_model()
                print(f"\nValidation Loss: {epoch_val_loss:.4f}")
                
                if args.wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "val_loss": epoch_val_loss
                    })
                
                # Save best model
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': best_val_loss,
                    }, best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                
                # Return to training mode
                model.train()

        progress_bar.close()

    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    print("Training the model")
    args = parse_args()
    train(args)