import math
import torch 
import torch.nn as nn 
import torch.nn.functional as  f 
from typing import Optional, Tuple
from model import scaled_dot_product 

class RMSNorm(nn.Module): 
    def __init__(self,dim:int): 
        self.dim = dim 
        self.gamma = nn.Parameter(torch.ones(self.dim))
    
    def forward(self,x):
        rms = torch.sqrt(torch.mean(x*x,dim=-1 ,keepdim=True))
        x = x / rms 
        return self.gamma * x 
        
def precompute_theta(dim: int, seq_len: int, theta: float = 10000):
    pass 

def apply_decoupled_rotary_embed(x: torch.Tensor, freqs_complex: torch.Tensor, d_rope: int):
    pass 

class MLP(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,dropout:float): 
        pass 


class MLA(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, max_seq_len=2048, d_rope=None):
        pass 


class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, hidden_dim, dropout, max_seq_len=2048,d_rope=None,ssmax=False):
        pass 


class MiniDeepseek(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, hidden_dim, dropout, num_layers, vocab_size, max_seq_len=2048,ssmax=False):
        pass 