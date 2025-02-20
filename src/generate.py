import torch 

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)  # Add batch dimension

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())  # Remove batch dimension




def generate(model, device, idx, max_new_tokens, context_len):
    idx = idx.to(device)  

    for _ in range(max_new_tokens):
        idx_context = idx[:, -context_len:]

        with torch.no_grad():
            idx_context = idx_context.to(device)
            logits = model(idx_context)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=-1)
    
    return idx