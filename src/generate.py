import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # Add batch dimension

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())  # Remove batch dimension

def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):

    unique_tokens = torch.unique(generated_tokens)
    logits[:, unique_tokens] = logits[:, unique_tokens] / penalty
    
    return logits

def generate(model, device, idx, max_new_tokens, context_len, repetition_penalty=1.2, temperature=1.0):
    idx = idx.to(device)
    
    for _ in range(max_new_tokens):
        # Get the context window
        idx_context = idx[:, -context_len:]
        
        with torch.no_grad():
            idx_context = idx_context.to(device)
            logits = model(idx_context)
            logits = logits[:, -1, :]  # Get predictions for next token
            
            # Apply repetition penalty
            penalized_logits = apply_repetition_penalty(
                logits.clone(),
                idx_context,
                penalty=repetition_penalty
            )
            
            # Apply temperature scaling
            if temperature != 1.0:
                penalized_logits = penalized_logits / temperature
            
            # Convert to probabilities
            probs = torch.softmax(penalized_logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with the running sequence
            idx = torch.cat((idx, idx_next), dim=-1)
    
    return idx