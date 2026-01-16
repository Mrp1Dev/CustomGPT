import torch
import myneural as mn
from myneural.module import Param as P

class Linear(mn.Module):
    
    def __init__(self, n_in, n_out, dtype = torch.float32, device = 'cpu', bias = False):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        self.weight = P(torch.randn(n_in, n_out, **kwargs) * (n_in ** -0.5)) # scaling down with * 1/sqrt(fan_in). gain isn't really required due to normalization layers.
        self.bias = P(torch.zeros(n_out, **kwargs)) if bias else None
        
    def forward(self, x):
        x = x @ self.weight # (B, T, C_in) @ (C_in, C_out) => (B, T, C_out)
        if self.bias is not None:
            x = x + self.bias # (B, T, C_out) + (T_broadcasted, C_out)
        return x
    
    def split(self, split_size, dim=0):
        """currently only for non bias version"""
        return self.weight.split(split_size, dim)

class Embedding(mn.Module):
    
    def __init__(self, n_in, n_out, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        self.embedding = P(torch.randn(n_in, n_out, **kwargs)) 
        
    def forward(self, x):
        return self.embedding[x] # (vocab_size, n_embd) indexed with (B, T) gives (B, T, n_embd) where values of x at [B, T] are within [0, vocab_size)
    
class SiLU(mn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) # modern replacement for reLU (precursor to swiGLU, might add that later. idk for small models though)
    
    
class LayerNorm(mn.Module):
    
    def __init__(self, n_in, eps = 1e-8,dtype = torch.float32, device = 'cpu'):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        self.eps = eps
        self.gamma = P(torch.ones(n_in, **kwargs)) # (n_embd, )
        self.beta = P(torch.zeros(n_in, **kwargs)) # (n_embd, ) but will be broadcasted to (n_embd_b, n_embd) during addition
    
    def forward(self, x):
        
        mean = x.mean(-1, keepdim = True) # (B, T, 1)
        var = x.var(-1, keepdim = True, unbiased = False) # (B, T, 1)
        x = (x - mean) * (var + self.eps)**-0.5 # ((B, T, C) - (B, T, C_b)) * (B, T, C_b) 
        x = self.gamma * x + self.beta # (B_b, T_b, C) * (B, T, C) + (B_b, T_b, C)
        return x
        
class MultiHeadAttention(mn.Module):
    def __init__(self, config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        assert config.n_embd % config.n_head == 0, "n_embd needs to be an integer multiple of n_heads."
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.qkv = Linear(config.n_embd, config.n_embd * 3, **kwargs) # q, k, v in a single matrix (C, 3*C)
        head_size = config.n_embd // config.n_head
        self.tril = torch.tril(torch.ones(config.context_length, config.context_length, **kwargs)).view(1,1,config.context_length, config.context_length) # so that later i can use this as (B_b, nh_b, T, T)
        # as this mask is applied to attention scores, which are a result of (B, nh, T, hs) @ (B, nh, hs, T) => (B, nh, T, T) 

    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv.split(self.n_embd, dim=-1) # split back along the (n_embd*3) dimension. each are ()
        head_size = self.n_embd // self.n_head # can use either self.n_embd or C here.
        q = q.view(1, 1, self.n_head, self.n_embd // self.n_head).transpose(1,2) # view (C,) as (B,T,nh,hs) => (B,nh,T,hs) where this B and T = 1 
        attention = (q @ k.T) * self.n_head**-0.5 # (B, nh, T, hs) @ (B, nh, hs, T) => (B, nh, T, T) 
        #TODO: finish attention
        
        
        