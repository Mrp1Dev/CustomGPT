import torch

def precompute_rotary_embeddings(head_size: int, context_length: int, theta: float = 10000.0, device = 'cpu'):

    assert head_size%2 == 0, "RoPE only supports even number of dimensions of q, k."
    powers = torch.arange(0, head_size, 2, dtype=torch.float32, device=device) / head_size # (0 to 1), steps of 2 cus we're pairing
    
    freqs = 1.0 / (theta ** powers)
    
    t = torch.arange(context_length,dtype=torch.float32,  device=device) 
    
    # outer product makes a matrix of size (context_length, hs//2) to get the angle for each position based on its corresponding frequency
    freqs = torch.outer(t, freqs)
    
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos[None,None,:,:], sin[None,None,:, :] # to be broadcasted later as (B_b, nh_b, T, hs//2)
    
    return cos, sin

def apply_rope(x, cos, sin):
    assert x.ndim == 4 # must be (B, nh, T, hs)
    x_fp32 = x.float()
    half = x_fp32.shape[-1]//2
    x1, x2 = x_fp32[..., :half], x_fp32[..., half:] # (B, nh, T, hs//2)
    # for a pair (x1, x2), where it'll be like (x1, x33) incase of dim=64, after rotating the new coords (X, Y):
    # its that table i learned in JEE, 
    # X = x1*cost - x2*sint | [cos, -sin]
    # Y = x1*sint + x2*cost | [sin,  cos]
    y1 = x1 * cos - x2 * sin # Y
    y2 = x1 * (sin) + x2 * cos # X
    # # cat along the last dimension (hs//2)
    return torch.cat([y1, y2], dim=3).type_as(x) # (B, nh, T, hs)


