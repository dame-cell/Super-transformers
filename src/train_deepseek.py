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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


from mini_deepseek import build_model 
from train import to_device

# Set random seed for reproducibility
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="context length for the model")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument('--train_data', type=str, help="path to the train npz file")
    parser.add_argument('--test_data', type=str, help="path to the test npz file")
    parser.add_argument('--wandb', action='store_true', help="Use Weights and Biases for logging")
    parser.add_argument('--ssmax', action='store_true', help="whether to use or not use scalable softmax")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size for training")
    parser.add_argument('--size', type=str, default="default", help="whether to use a small or default or large architecture ")
    parser.add_argument('--generate', type=int, default=300, help="what step to generate during training")
    parser.add_argument('--train_log', type=int, default=100, help="what step to log the train loss during training")
    parser.add_argument('--grad_accumulation_steps', type=int, default=2, help="Grad accumlation")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")

    return parser.parse_args()

def generate_samples(model, tokenizer, device, num_samples=3, max_length=50, temperature=0.8):
    """Generate and display sample outputs from the model."""
    model.to(device)
    model.eval()

    # Set different prompt texts
    prompts = [
        "As an AI Language Model,I ",
        "Today at school, I learned about"
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


def train(args): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb:
        wandb.login()  # Removed hardcoded API key for security
        wandb.init(project="mini-deepseek", config=vars(args))

    model = build_model(size=args.size, ssmax=args.ssmax, max_seq_len=args.max_seq_len)
    model.to(device)

    train_data = GPTDatasetV1(args.train_data)
    val_data = GPTDatasetV1(args.test_data)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    global_step = 0


    no_decay = ["bias", "rmsnorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    num_update_steps_per_epoch = len(train_loader) // args.grad_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch

    # Learning rate scheduler using transformers' get_cosine_schedule_with_warmup
    print("Setting up transformers cosine warmup scheduler...")
    warmup_steps = int(0.1 * max_train_steps)  # 10% of training for warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )

    for epoch in range(args.epochs): 
        model.train()
        epoch_loss = 0 
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        optimizer.zero_grad()

        for step, (input_batch, target_batch) in enumerate(train_pbar):
            input_batch, target_batch = to_device(input_batch, device), to_device(target_batch, device)
            logits = model(input_batch)
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_batch.view(-1),
                label_smoothing=0.1
            )

            loss = loss / args.grad_accumulation_steps
            loss.backward()
            if (step + 1) % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Track loss
            current_loss = loss.item() * args.grad_accumulation_steps
            epoch_loss += current_loss

            # Update progress bar with current loss and learning rate
            current_lr = lr_scheduler.get_last_lr()[0]
            train_pbar.set_postfix({"loss": current_loss, "lr": current_lr})

            # Log to wandb
            if args.wandb and step % args.train_log == 0:
                wandb.log({
                    "train_loss": current_loss,
                    "learning_rate": current_lr,
                    "global_step": global_step
                })

            if args.wandb and step % args.generate == 0:
                generate_samples(model, tokenizer, device)

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")

        with torch.no_grad():
            for step, (input_batch, target_batch) in enumerate(val_loader):
                input_batch, target_batch = to_device(input_batch, device), to_device(target_batch, device)
                logits = model(input_batch)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_batch.view(-1),
                    label_smoothing=0.1  
                )

                val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log epoch results
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss_epoch": avg_train_loss,
                "val_loss_epoch": avg_val_loss
            })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "llama_mla_best_model.pt")
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    print("Training the model")
    args = parse_args()
    train(args)