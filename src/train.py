import torch 
import gzip 
import wandb 
import argparse  
import numpy as np 
import torch.nn as nn 
from tqdm.auto import tqdm 
from model import build_model
import torch.nn.functional as F
from utils import TextSamplerDataset 
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--max_len', type=int, default=1024, help="context length for the model")
    parser.add_argument('--epoch', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--wandb', type=bool, default=False, help="Use Weights and Biases for logging")
    parser.add_argument('--ssmax', type=bool, default=True, help="whether to use or not use scalable softmax")
    parser.add_argument('--use_pos_enc', type=bool, default=False, help="whether to use or not use Positional encodings")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size for training")
    parser.add_argument('--size', type=str, default="default", help="whether to use a small or default or large architecture ")

    return parser.parse_args()

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)  # Faster transfer with non_blocking=True
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data  # If it's not a tensor, return as is

def decode_token(token):
    if torch.is_tensor(token):
        # Handle single token only
        if token.numel() == 1:
            token = token.item()
        else:
            raise ValueError("decode_token expects a single token")
    return str(chr(max(32, int(token))))

def decode_tokens(tokens):
    if torch.is_tensor(tokens):
        # Convert the entire tensor to a list of integers
        tokens = tokens.cpu().numpy().tolist()
    return ''.join(map(decode_token, tokens))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="super-transformers", config=args)

    model = build_model(size=args.size, ssmax=args.ssmax, use_pos_enc=args.use_pos_enc)
    model.to(device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable CuDNN optimization

    with gzip.open("enwik8.gz", "rb") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    train_dataset = TextSamplerDataset(data_train, args.max_len)
    val_dataset   = TextSamplerDataset(data_val, args.max_len)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)

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
            
   
          

            if step % 1000 == 0 and step > 0:
                model.eval()
                import random 
                sample = random.choice(val_dataset)
                inp = sample[0][:-1]
                inp = inp.unsqueeze(0) if inp.dim() == 1 else inp
                inp = to_device(inp, device)
                
                try:
                    prime = decode_tokens(inp[0])
                    print(f'\nInput text:\n{prime}\n\n{"*" * 100}')
                    
                    with torch.no_grad():
                        full_sequence = model.generate(
                            input_ids=inp,
                            max_length=inp.size(1) + 100,  # Generate 100 new tokens
                            temperature=0.8
                        )
                        
                        
                        generated_text = decode_tokens(full_sequence[0, inp.size(1):])
                        print(f"\nGenerated continuation:", generated_text)

                except Exception as e:
                    print(f"Error during generation: {str(e)}")
                
                model.train()

            if step % 100 == 0 and step > 0:
                step_loss = running_loss / (step + 1)  # Correct loss calculation
                print(f"Step {step+1} - Train Loss: {step_loss:.4f}")
    
                if args.wandb:
                    wandb.log({"step": step + 1, "train_loss": step_loss})  # Log step loss correctly


        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epoch} - Train Loss: {epoch_loss:.4f}")

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

