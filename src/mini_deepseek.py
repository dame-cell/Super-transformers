import math
import torch 
import torch.nn as nn 
import torch.nn.functional as  F 
from model import scaled_dot_product 

class RMSNorm(nn.Module): 
    def __init__(self,dim:int): 
        super().__init__()
        self.dim = dim 
        self.gamma = nn.Parameter(torch.ones(self.dim))
    
    def forward(self,x):
        rms = torch.sqrt(torch.mean(x*x,dim=-1 ,keepdim=True))
        x = x / rms 
        return self.gamma * x 
        
def precompute_theta(dim: int, seq_len: int, theta: float = 10000):
    assert  dim % 2 == 0 , "must be divisble by 2"
    # get the numerator 
    theta_num = torch.arange(0,dim,2) 
    # calclate the actual theta 
    theta_values = 1.0 / theta ** (theta_num / dim )
    # compute the position ids 
    m =  torch.arange(seq_len)
    # Compute frequency matrix by outer product of positions and theta
    # Outer product: (seq_len,) ⊗ (dim/2,) → (seq_len, dim/2)
    freqs = torch.outer(m,theta_values).float()
    # Compute complex number representation using polar form (r * exp(iθ))
    # Since r = 1, we only use θ values computed above
    # Shape after: (seq_len, dim/2) as complex numbers
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex



def apply_decoupled_rotary_embed(x: torch.Tensor, freqs_complex: torch.Tensor, d_rope: int):
    # Extract the dimensions of the input tensor
    bs, seq_len, n_heads, head_dim = x.size()  # batch size, sequence length, number of heads, head dimension

    # Split the tensor into two parts:
    # - `x_rope`: The first `d_rope` dimensions where RoPE is applied.
    # - `x_base`: The remaining dimensions that are not affected by RoPE.
    x_rope = x[:, :, :, :d_rope]  # (bs, seq_len, n_heads, d_rope)
    x_base = x[:, :, :, d_rope:]  # (bs, seq_len, n_heads, head_dim - d_rope)

    # Reshape `x_rope` to prepare it for complex number operations.
    # We treat every **two consecutive** elements as a real and imaginary part of a complex number.
    # The `-1` automatically calculates the required size.
    x_rope_reshape = x_rope.float().contiguous().reshape(bs, seq_len, n_heads, -1, 2)

    # Convert real-valued tensors to complex numbers, where:
    # - `[..., 0]` corresponds to the real part.
    # - `[..., 1]` corresponds to the imaginary part.
    x_rope_complex = torch.view_as_complex(x_rope_reshape)  # Shape: (bs, seq_len, n_heads, d_rope/2)

    # Truncate `freqs_complex` to match `d_rope/2`, ensuring we only apply rotation to the correct subspace.
    # `unsqueeze` ensures proper broadcasting across the batch and head dimensions.
    freqs_complex_truncated = freqs_complex[:seq_len, :d_rope//2].unsqueeze(0).unsqueeze(2)  
    # Shape: (1, seq_len, 1, d_rope/2)

    # Apply RoPE by multiplying the complex vectors with the corresponding rotational frequencies.
    # This effectively rotates the embedding vectors in the complex plane.
    x_rope_rotated = x_rope_complex * freqs_complex_truncated  # Shape: (bs, seq_len, n_heads, d_rope/2)

    # Convert the rotated complex tensor back to a real-valued tensor.
    # The output is a tensor with the first half as real parts and the second half as imaginary parts.
    x_rope_out_real = torch.view_as_real(x_rope_rotated)  # Shape: (bs, seq_len, n_heads, d_rope/2, 2)

    # Reshape it back to match the original `x_rope` shape before transformation.
    x_rope_out = x_rope_out_real.reshape(bs, seq_len, n_heads, d_rope)  # Shape: (bs, seq_len, n_heads, d_rope)

    # Concatenate the RoPE-transformed part (`x_rope_out`) and the untouched part (`x_base`).
    # The result has the same shape as the input tensor, with RoPE applied only to `d_rope` dimensions.
    return torch.cat([x_rope_out, x_base], dim=-1)  # Shape: (bs, seq_len, n_heads, head_dim)

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)  # Linear projection for SwiGLU input
        self.w2 = nn.Linear(dim, hidden_dim)  # Linear projection for SwiGLU gating mechanism
        self.w3 = nn.Linear(hidden_dim, dim)  # Output projection back to original dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
  
        x_gate = self.w1(x)  # Main transformation
        x_gate = nn.functional.silu(x_gate)  # Apply SiLU (Swish) activation
        x_out = self.w2(x)  # Gating mechanism
        x = x_gate * x_out  # Element-wise multiplication (SwiGLU)
        x = self.w3(x)  # Project back to original dimension
        x = self.dropout(x)  # Apply dropout
        return x

class MOE(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,dropout:float):
        pass 
        
class MLA(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, max_seq_len=2048, d_rope=None):
        super().__init__()
        self.dim = dim 
        self.num_heads = num_heads 
        self.max_seq_len = max_seq_len
        self.head_dim = dim // num_heads 
        self.latent_dim = latent_dim 

        self.d_rope = d_rope if d_rope is not None else self.head_dim // 4
        assert self.d_rope % 2 == 0, "d_rope must be divisible by 2 for complex representation"
        assert self.d_rope <= self.head_dim, "d_rope must be less than or equal to head_dim"
        
        # Query projection 
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        
        # Latent key-value projections (for compression)
        self.wk_a = nn.Linear(dim, latent_dim, bias=False)
        self.wk_b = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        
        self.wv_a = nn.Linear(dim, latent_dim, bias=False)
        self.wv_b = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Scale params for the scaled softmax 
        self.scale_param = nn.Parameter(torch.ones(num_heads))
        
        # Cache for k and v 
        self.cache_k = None 
        self.cache_v = None 
        
        # Flag to track if weights have been absorbed for inference 
        self.weights_absorbed = False
        
        # For optimized inference path
        self.wk = None
        self.wv = None
    
    def absorb_weights(self):
        """
        Absorb the two-step key and value projections into single matrices for inference.
        This is an optimization that reduces computation during inference.
        """
        if not self.weights_absorbed:
            device = self.wk_a.weight.device  # Get the device from an existing parameter
            
            # Create new linear layers for the absorbed weights on the same device
            self.wk = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False, device=device)
            self.wv = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False, device=device)
            
            # Compute the matrix multiplication of the two-step projections
            with torch.no_grad():
                self.wk.weight.copy_(self.wk_b.weight @ self.wk_a.weight)
                self.wv.weight.copy_(self.wv_b.weight @ self.wv_a.weight)
            
            # Mark as absorbed
            self.weights_absorbed = True
    
    def forward(self, x, mask=None, start_pos=0, freqs_complex=None, ssmax=False, use_cache=True):
        bs, seq_len, dim = x.size()
        device = x.device  # Get the device from the input tensor
        
        # Initialize cache if needed
        if use_cache and (self.cache_k is None or self.cache_k.size(0) != bs):
            self.cache_k = torch.zeros(
                (bs, self.num_heads * self.head_dim, self.max_seq_len),
                device=device, dtype=x.dtype
            )
            self.cache_v = torch.zeros(
                (bs, self.num_heads * self.head_dim, self.max_seq_len),
                device=device, dtype=x.dtype
            )
        
        # Current sequence length including previous context
        curr_len = start_pos + seq_len
        
        # Project queries
        q = self.wq(x).view(bs, seq_len, self.num_heads, self.head_dim)
        
        # Handle key and value projections based on whether weights are absorbed
        if self.weights_absorbed and self.wk is not None and self.wv is not None:
            # Use the absorbed projection matrices for inference
            k_curr = self.wk(x).view(bs, seq_len, self.num_heads, self.head_dim)
            v_curr = self.wv(x).view(bs, seq_len, self.num_heads, self.head_dim)
        else:
            # Use the two-step projection for training
            k_latent = self.wk_a(x)
            v_latent = self.wv_a(x)
            
            k_curr = self.wk_b(k_latent).view(bs, seq_len, self.num_heads, self.head_dim)
            v_curr = self.wv_b(v_latent).view(bs, seq_len, self.num_heads, self.head_dim)
        
        # Handle caching for autoregressive generation
        if use_cache and start_pos > 0:
            # Flatten the head dimension for caching
            k_curr_flat = k_curr.reshape(bs, seq_len, -1)
            v_curr_flat = v_curr.reshape(bs, seq_len, -1)
            
            # Update cache with current tokens
            self.cache_k[:bs, :, start_pos:start_pos+seq_len] = k_curr_flat.permute(0, 2, 1)
            self.cache_v[:bs, :, start_pos:start_pos+seq_len] = v_curr_flat.permute(0, 2, 1)
            
            # Retrieve the full cached sequence (including previous tokens)
            k_flat = self.cache_k[:bs, :, :curr_len].permute(0, 2, 1)
            v_flat = self.cache_v[:bs, :, :curr_len].permute(0, 2, 1)
            
            # Reshape to include head dimension
            k = k_flat.view(bs, curr_len, self.num_heads, self.head_dim)
            v = v_flat.view(bs, curr_len, self.num_heads, self.head_dim)
        else:
            # For the first token or when not using cache
            k = k_curr
            v = v_curr
        
        # Prepare tensors for attention computation
        q = q.permute(0, 2, 1, 3)  # [bs, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)  # [bs, num_heads, curr_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [bs, num_heads, curr_len, head_dim]
        
        # Apply rotary embeddings if provided
        if freqs_complex is not None:
            # Ensure freqs_complex is on the same device
            if freqs_complex.device != device:
                freqs_complex = freqs_complex.to(device)
            q = apply_decoupled_rotary_embed(q, freqs_complex, self.d_rope)
            k = apply_decoupled_rotary_embed(k, freqs_complex, self.d_rope)

        # Compute attention with or without scaled softmax
        if ssmax:
            out, attention = scaled_dot_product(q, k, v, mask, self.scale_param)
        else:
            out, attention = scaled_dot_product(q, k, v, mask)
        
        # Reshape output and project
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, self.num_heads * self.head_dim)
        return self.wo(out)
        
class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, hidden_dim, dropout, max_seq_len=2048,d_rope=None,ssmax=False):
        super().__init__()
        self.rmsnorm1 = RMSNorm(dim)
        self.attention = MLA(dim,num_heads,latent_dim,max_seq_len,d_rope)
        self.rmsnorm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim,hidden_dim,dropout)

        self.ssmax = ssmax  

    def forward(self, x, mask=None, start_pos=0, freqs_complex=None, use_cache=True):
        residual = x
        x = self.rmsnorm1(x)
        
        attn_out = self.attention(
            x,
            mask=mask,
            start_pos=start_pos,
            freqs_complex=freqs_complex,
            ssmax=self.ssmax,
            use_cache=use_cache
        )
        
        x = residual + attn_out
        residual = x
        x = self.rmsnorm2(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out
        return x

class MiniDeepseek(nn.Module):
    def __init__(self, dim, num_heads,hidden_dim, dropout, num_layers, vocab_size, max_seq_len=2048,ssmax=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Calculate latent dimension for MLA based on num_kv_heads
        self.latent_dim = dim // num_heads

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.ssmax = ssmax
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                dim=dim,
                num_heads=num_heads,
                latent_dim=self.latent_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                max_seq_len=max_seq_len,
                ssmax = self.ssmax,
                d_rope=dim // (4 * num_heads)  # Default RoPE dimension
            ) for _ in range(num_layers)
        ])

        # Final normalization and output projection
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Initialize rotary embeddings
        self.freqs_complex = None
        self.initialize_rotary_embeddings()

        # Apply weight tying
        self.token_embeddings.weight = self.output.weight

    def initialize_rotary_embeddings(self):
        self.freqs_complex = precompute_theta(
            dim=self.dim // self.num_heads,
            seq_len=self.max_seq_len
        )

    def get_parameter_count(self):
        """
        Get the total number of parameters in the model

        Returns:
            total_params: Total number of parameters
            trainable_params: Number of trainable parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text autoregressively

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling

        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()  # Set to evaluation mode

        # Initialize with input_ids
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()

        # Clear layer caches
        for layer in self.layers:
            layer.attention.cache_k = None
            layer.attention.cache_v = None

        # Generate tokens autoregressively
        for i in range(max_new_tokens):
            # Get the last token if we're beyond the first iteration
            if i > 0:
                curr_input_ids = generated_ids[:, -1].unsqueeze(-1)
                start_pos = generated_ids.shape[1] - 1
            else:
                curr_input_ids = generated_ids
                start_pos = 0

            # Forward pass
            with torch.no_grad():
                logits = self.forward(
                    curr_input_ids,
                    start_pos=start_pos,
                    use_cache=True,
                    #ssmax=True  # Use scalable softmax for generation
                )

            # Get the logits for the last token
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated_ids
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return generated_ids

    def absorb_weights_for_inference(self):
        """
        Absorb weights for more efficient inference by merging matrices
        where possible in the MLA attention layers.
        """
        for layer in self.layers:
            layer.attention.absorb_weights()

        print("Weights absorbed for efficient inference.")

    def forward(self, input_ids, start_pos=0, use_cache=True):

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create causal mask for autoregressive attention
        # During training or for the first token, we need the full mask
        if not use_cache or start_pos == 0:
            # Full causal mask for training
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        else:
            # For generation with caching, we only need to mask the current token's attention
            # to previously generated tokens
            curr_len = start_pos + seq_len
            mask = torch.zeros(seq_len, curr_len, device=device, dtype=torch.bool)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, curr_len]

        # Get embeddings
        x = self.token_embeddings(input_ids)  # [batch_size, seq_len, dim]

        # Make sure we have the right device for freqs_complex
        if self.freqs_complex.device != device:
            self.freqs_complex = self.freqs_complex.to(device)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(
                x,
                start_pos=start_pos,
                mask=mask,
                freqs_complex=self.freqs_complex,
                use_cache=use_cache,
            )

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output(x)

        return logits

def build_model(size: str = "default", ssmax: bool = True, max_seq_len:int=2048):
    """
    Build a MiniDeepseek model in different sizes.
    
    Args:
        size (str): Size of the model - "small", "default", or "large"
        ssmax (bool): Whether to use scaled softmax, default is True
    
    Returns:
        MiniDeepseek: Model instance
    """
    # GPT-2 vocabulary size
    vocab_size = 50257
    max_seq_len = max_seq_len
    
    # Model configurations for different sizes
    if size.lower() == "small":
        model = MiniDeepseek(
            dim=512,
            num_heads=8,
            hidden_dim=1376,  # ~2.7x dimension for SwiGLU
            dropout=0.1,
            num_layers=6,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            ssmax=ssmax
        )
    elif size.lower() == "default":
        model = MiniDeepseek(
            dim=768,
            num_heads=12,
            hidden_dim=2048,
            dropout=0.1,
            num_layers=12,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            ssmax=ssmax
        )
    elif size.lower() == "large":
        model = MiniDeepseek(
            dim=1024,
            num_heads=16,
            hidden_dim=2816,  # ~2.75x dimension
            dropout=0.1,
            num_layers=24,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            ssmax=ssmax
        )
    else:
        raise ValueError(f"Unsupported model size: {size}. Choose 'small', 'default', or 'large'.")
    
    # Print model size information
    param_info = model.get_parameter_count()
    print(f"Built MiniDeepseek-{size} model")
    print(f"Parameters: {param_info['total_params'] / 1e6:.2f}M")
    print(f"Trainable parameters: {param_info['trainable_params'] / 1e6:.2f}M")
    
    return model

