import math
import time
import wandb  
import torch
import argparse 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils import GPTDatasetV1
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


from mini_deepseek import build_model 
from train import to_device

# Set random seed for reproducibility
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
 
def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="context length for the model")
    parser.add_argument('--epoch', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument('--train_data', type=str, help="path to the train npz file")
    parser.add_argument('--test_data', type=str, help="path to the test npz file")
    parser.add_argument('--wandb', action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument('--ssmax', action='store_true', help="whether to use or not use scalable softmax")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size for training")
    parser.add_argument('--size', type=str, default="small", help="whether to use a small or default or large architecture ")
    parser.add_argument('--generate', type=int, default=300, help="what step to generate during training")
    parser.add_argument('--log_interval', type=int, default=100, help="what step to log losses during training")
    parser.add_argument('--validation_step', type=int, default=100, help="what step to log val losses during training")

    return parser.parse_args()

def generate_samples(model, tokenizer, device, num_samples=3, max_length=50, temperature=0.8):
    """Generate and display sample outputs from the model."""
    model.to(device)
    model.eval()

    # Set different prompt texts
    prompts = [
        "Once upon a time,",
    ]

    for i, prompt in enumerate(prompts[:num_samples]):
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.9
            )

        # Decode and print the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Sample {i+1}:\nPrompt: {prompt}\nGenerated: {generated_text}\n")


def validate(model, val_loader, device, tokenizer):
    """Run validation on the entire validation set and return average loss."""
    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for input_batch, target_batch in val_pbar:
            input_batch, target_batch = to_device(input_batch, device), to_device(target_batch, device)
            logits = model(input_batch)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_batch.view(-1),
                label_smoothing=0.1,
                ignore_index=tokenizer.pad_token_id
            )
            val_loss += loss.item()
            val_pbar.set_postfix({"loss": loss.item()})
    
    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project="mini_deepseek", config=args)

    # Model setup
    model = build_model(size=args.size, ssmax=args.ssmax, max_seq_len=args.max_seq_len)
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
    import os 
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

            # Gradient trackingcl
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
            if step % args.log_interval == 0 and step > 0:
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
            if step % args.generate == 0 and step > 0:
                sample_text = generate_samples(model, tokenizer, device,num_samples=1)
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
