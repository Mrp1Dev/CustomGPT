import torch
import torch.nn as nn
import myneural as mn
from dataclasses import dataclass

@dataclass
class Config:
    n_embd = 192
    n_head = 6
    context_length = 256
    vocab_size = 65
    n_layer = 6
    batch_size = 64
    device = 'cuda'


class Block(nn.Module):
    def __init__(self, config: Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.attention = mn.MultiHeadAttention(config, dtype, device)
        self.feed_forward = mn.FeedForward(config, dtype, device)
        self.ln1 = mn.LayerNorm(config.n_embd, 1e-8, dtype, device)
        self.ln2 = mn.LayerNorm(config.n_embd, 1e-8, dtype, device)
        
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.config = config
        self.token_embedding = mn.Embedding(config.vocab_size, config.n_embd, dtype, device)
        self.pos_embedding = mn.Embedding(config.context_length, config.n_embd, dtype, device)
        self.transformer = mn.Sequential([
            Block(config, dtype, device) for _ in range(config.n_layer)
        ])
        self.ln_final = mn.LayerNorm(config.n_embd, 1e-8, dtype, device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding(idx) # (B, T) => (B, T, C)
        pos_embedding = self.pos_embedding(torch.arange(0, T, device = idx.device)) # (T,C)
        x = token_embedding + pos_embedding # (B, T, C) + (B_b, T, C)
        x = self.transformer(x)
        x = self.ln_final(x)
        logits = x @ self.token_embedding.embedding.T # lm_head, but directly applied (B, T, C) @ (C, vocab_size) => (B, T, vocab_size)
        if targets is not None:
            logits_flat = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets)
        else: 
            loss = torch.zeros((1))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cut = idx[:, -self.config.context_length:]
            logits, loss = self(idx_cut) # (B, T, C)
            logits = logits[:, -1, :] # last token, (B, C)
            probs = torch.softmax(logits, dim = -1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_new], dim=-1) 
        return idx
            