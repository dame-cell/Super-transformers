import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
# This code has been heavily inspired from huggingface ,they way they code and their structure 


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal mask for autoregressive generation."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.to(device)

def scaled_dot_product(q: torch.Tensor, 
                      k: torch.Tensor, 
                      v: torch.Tensor, 
                      mask: Optional[torch.Tensor] = None, 
                      scale_param: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced scaled dot product attention with optional SSMax scaling.
    Args:
        q, k, v: Query, Key, Value tensors
        mask: Attention mask
        scale_param: Optional learnable scaling parameter for SSMax
    """
    d_k = q.size(-1)
    
    if scale_param is not None:
        # Get context size and apply SSMax scaling
        n = k.size(-2)
        scale = scale_param.unsqueeze(-1) * torch.log(torch.tensor(n, dtype=q.dtype, device=q.device))
        #scale = scale.clamp(min=1e-5)  # Prevent numerical instability
        q = scale.unsqueeze(-1) * q
    
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    attention = F.softmax(scores, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, ssmax: bool = False, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_layer = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize learnable scaling parameter for SSMax
        self.scale_param = nn.Parameter(torch.ones(num_heads)) if ssmax else None
        
        # Initialize attention entropy tracking
        self.register_buffer('attention_entropy', torch.zeros(1))
        
    def compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions.So I can check the attention distributions."""
        entropy = -(attention * torch.log(attention + 1e-9)).sum(dim=-1).mean()
        return entropy
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        values, attention = scaled_dot_product(q, k, v, mask, self.scale_param)
        
        # Track attention entropy
        self.attention_entropy = self.compute_attention_entropy(attention)
        
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
        return self.dropout(self.out_proj(values))

class MultiHeadLatentAttention(nn.Module):
    def __init__(self):
        pass 

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()  # Changed to GELU for better performance
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, ssmax: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, ssmax, dropout)
        self.self_latent_attn = MultiHeadLatentAttention()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,latent_attention:bool=False) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        normalized = self.norm1(x)
        if latent_attention:
                attn_out = self._latent_attn(normalized,mask)
        attn_out = self.self_attn(normalized, mask)
        x = x + self.dropout(attn_out)
        
        normalized = self.norm2(x)
        mlp_out = self.mlp(normalized)
        x = x + self.dropout(mlp_out)
        return x

class GPT2Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 dim: int,
                 num_heads: int,
                 num_layers: int,
                 ffn_dim: int,
                 max_len: int = 1024,
                 dropout: float = 0.1,
                 ssmax: bool = False,
                 use_pos_enc: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_encoding = PositionalEncoding(dim, max_len, dropout)
            
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, num_heads, ffn_dim, dropout, ssmax)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    # initialize weights 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def get_attention_entropy(self) -> torch.Tensor:
        """Return average attention entropy across all layers."""
        return torch.mean(torch.stack([layer.self_attn.attention_entropy for layer in self.layers]))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Create causal mask if none provided
        if mask is None:
            mask = create_causal_mask(x.size(1), x.device)
            
        x = self.embedding(x)
        if self.use_pos_enc:
            x = self.pos_encoding(x)
            
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.lm_head(x)
    
        
def build_model(size: str, ssmax: bool = True, use_pos_enc: bool = False):
    config = {
        "small": {"vocab_size": 50257, "dim": 768, "num_heads": 8, "num_layers": 6, "ffn_dim": 2048, "max_len": 1024, "dropout": 0.1},

        "default": {
            "vocab_size": 50257, 
            "dim": 1024, 
            "num_heads": 16, 
            "num_layers": 24, 
            "ffn_dim": 4096, 
            "max_len": 2048, 
            "dropout": 0.1
        },
        "large": {
            "vocab_size": 50257, 
            "dim": 1280, 
            "num_heads": 20, 
            "num_layers": 36, 
            "ffn_dim": 5120, 
            "max_len": 4096, 
            "dropout": 0.1
        },
    }

    if size not in config:
        raise ValueError("Please choose between 'small', 'default', and 'large' for size.")
    
    print(f"We are using model size: {size}")
    return GPT2Decoder(
        vocab_size=config[size]['vocab_size'],
        dim=config[size]['dim'],
        num_heads=config[size]['num_heads'],
        num_layers=config[size]['num_layers'],  
        ffn_dim=config[size]['ffn_dim'],
        max_len=config[size]['max_len'],
        dropout=config[size]['dropout'],
        ssmax=ssmax,
        use_pos_enc=use_pos_enc  
    )
    
def test_model():
    """Test the model with example inputs."""
    # Model parameters
    vocab_size = 50257
    dim = 768
    num_heads = 12
    num_layers = 12
    ffn_dim = 3072
    max_len = 1024
    dropout = 0.1
    ssmax = True
    use_pos_enc = False
    
    # Initialize model
    model = GPT2Decoder(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        max_len=max_len,
        dropout=dropout,
        ssmax=ssmax,
        use_pos_enc=use_pos_enc
    )
    
    # Create example input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    output = model(input_ids)
    print(f"Output shape: {output.shape}")
    
    # Test generation
    generated = model.generate(
        input_ids=input_ids[:, :1],  # Start with first token
        max_length=16,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    print(f"Generated sequence shape: {generated.shape}")
    
    # Print attention entropy
    print(f"Average attention entropy: {model.get_attention_entropy().item():.4f}")

if __name__ == "__main__":
    test_model()
