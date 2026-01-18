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

class Embedding(mn.Module):
    
    def __init__(self, n_in, n_out, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        self.embedding = P(torch.randn(n_in, n_out, **kwargs) * n_out**-0.5) 
        
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
       
class RMSNorm(mn.Module):
    
    def __init__(self, n_in, eps = 1e-8,dtype = torch.float32, device = 'cpu'):
        super().__init__()
        kwargs = {'dtype': dtype, 'device': device}
        self.eps = eps
        self.gamma = P(torch.ones(n_in, **kwargs)) # (n_embd, )
        # no beta!
    
    def forward(self, x: torch.Tensor):
        
        # RMS: square, then mean, then root. eg. sqrt((x2 + y2 + z2)/3)
        x_fp32 = x.float() # we cast to fp32 so that when we square and mean etc we have higher accuracy in such large values
        square = x_fp32.pow(2) # (B, T, C) all values of all tokens in all batches squared
        mean = square.mean(dim=-1, keepdim=True) # (B, T, 1) all embeddings for each token of each batch are mean'd. 
        # we keepdim so that when i divide later we don't have the issue of (B, T, C) / (B, T) [would get C vs T not equal issue]
        root_reciprocal = torch.rsqrt(mean + self.eps) # r_sqrt is just faster for reciprocals
        
        return self.gamma * (x_fp32 * root_reciprocal).type_as(x) # gamma*(x/rms(x)), type is casted back to normal (bf16 probably)      
        
class MultiHeadAttention(mn.Module):
    def __init__(self, config: mn.Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.kwargs = {'dtype': dtype, 'device': device}
        assert config.n_embd % config.n_head == 0, "n_embd needs to be an integer multiple of n_heads."
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.qkv = Linear(config.n_embd, config.n_embd * 3, **self.kwargs) # q, k, v in a single matrix (C, 3*C)
        self.tril = torch.tril(torch.ones(config.context_length, config.context_length, **self.kwargs)).view(1,1,config.context_length, config.context_length)
        # so that later i can use this as (B_b, nh_b, T, T)
        # as this mask is applied to attention scores, (B, nh, T, T) 
        self.out_proj = Linear(config.n_embd, config.n_embd, **self.kwargs)
        

    def forward(self, x):
        
        B,T,C = x.shape # C == n_embd
        qkv = self.qkv(x) # gives (B, T, C) @ (C, C*3) => (B, T, C*3)
        q,k,v = torch.split(qkv, self.n_embd, dim=-1) # split back along the (C*3) to get (B,T,C)
        head_size = self.n_embd // self.n_head # head_size * n_head = n_embd
        q = q.view(B, T, self.n_head, head_size).transpose(1,2) # view (B,T,C) as (B,T,nh,hs) => (B,nh,T,hs) where nh*hs = C 
        k = k.view(B, T, self.n_head, head_size).transpose(1,2) # (B,nh,T,hs) 
        v = v.view(B, T, self.n_head, head_size).transpose(1,2) # (B,nh,T,hs) 
        # attention = (q @ k.transpose(-2, -1)) * head_size**-0.5 # (B, nh, T, hs) @ (B, nh, hs, T) => (B, nh, T, T) 
        # attention = attention.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf')) # the == 0 creates a boolean mask of (1, 1, T, T)
        # attention = torch.softmax(attention, dim = -1) # we want each row in (T,T) to sum to 1 as it acts like probabilities of how useful other tokens are for each token
        # result = attention @ v # (B, nh, T, T) @ (B, nh, T, hs) => (B, nh, T, hs) each token focuses on highly aligned other tokens to weigh in their embeddings
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v, self.tril)
        result = result.transpose(1,2).contiguous().view(B, T, C) # (B, nh, T, hs) => (B, T, nh, hs), make them contiguous so we can view them as (B, T, C)
        result = self.out_proj(result) # (B, T, C) @ (C, C) => (B, T, C)
        return x
        
class Sequential(mn.Module):
    def __init__(self, layers: list) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
class FeedForward(mn.Module):
    def __init__(self, config: mn.Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.kwargs = {'dtype': dtype, 'device': device}
        self.mlp = Sequential([
            Linear(config.n_embd, config.n_embd * 4, **self.kwargs),
            SiLU(),
            Linear(config.n_embd*4, config.n_embd, **self.kwargs)
        ])
    
    def forward(self, x):
        x = self.mlp(x)
        return x