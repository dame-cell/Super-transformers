import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ScalableSoftmax(nn.Module):
    def __init__(self, scale_factor=0.2):
        super(ScalableSoftmax, self).__init__()
        self.scale_factor = scale_factor  

    def forward(self, x):
        """
        Applies Scalable Softmax (SSMax) to the input tensor.
        The formula is Softmax(x * s), where s is the scaling factor.

        Parameters:
            x (torch.Tensor): The input tensor to apply Scalable Softmax on.

        Returns:
            torch.Tensor: The output after applying Scalable Softmax.
        """
        scaled_x = self.scale_factor * x
        
        exp_x = torch.exp(scaled_x)
        softmax_output = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
        
        return softmax_output


def scaled_dot_product(q, k, v, mask=None, ssmax=False):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    if ssmax:
        attention = ScalableSoftmax()(scores)
    else:
        attention = F.softmax(scores, dim=-1)
    
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, ssmax=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_layer = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ssmax = ssmax
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv_layer(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask, self.ssmax)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
        return self.out_proj(values)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1, ssmax=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, ssmax)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x

class GPT2Decoder(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, ffn_dim, max_len=1024, dropout=0.1, ssmax=False, use_pos_enc=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_encoding = PositionalEncoding(dim, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, num_heads, ffn_dim, dropout, ssmax) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        if self.use_pos_enc:
            x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

if __name__ == "__main__":
 
    # Set parameters for GPT-2
    vocab_size = 50257  # Common GPT-2 vocab size
    dim = 768  # Dimension of model embeddings (GPT-2 uses 768 by default for base models)
    num_heads = 12  # Number of attention heads
    num_layers = 12  # Number of transformer layers
    ffn_dim = 3072  # Feed-forward network dimension, typically 4 times dim
    max_len = 1024  # Max input sequence length
    dropout = 0.1  # Dropout rate
    ssmax = True  # Use Scalable Softmax
    use_pos_enc = False  # Use positional encoding

    # Instantiate the GPT-2 decoder model
    model = GPT2Decoder(vocab_size, dim, num_heads, num_layers, ffn_dim, max_len, dropout, ssmax, use_pos_enc)

    # Create dummy input sequence of tokenized text (e.g., [50256] is often the EOS token in GPT-2 vocab)
    input_seq = torch.randint(0, vocab_size, (1, 10))  # Batch size of 1, sequence length of 10 tokens

    # Perform a forward pass through the model
    output = model(input_seq)

    # Print the output (logits for each token in the sequence)
    print(output.shape)  # Output shape: (batch_size, sequence_length, vocab_size)
    print(output)
