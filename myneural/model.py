import torch
import myneural as mn
from sentencepiece import SentencePieceProcessor as sp
class Block(mn.Module):
    def __init__(self, config: mn.Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.attention = mn.MultiHeadAttention(config, dtype, device)
        self.feed_forward = mn.SwiGLU(config, dtype, device) # i've updated from linear MLP to SwiGLU, but its still considered an ffwd layer.
        self.rmsn_1 = mn.RMSNorm(config.n_embd, 1e-8, dtype, device)
        self.rmsn_2 = mn.RMSNorm(config.n_embd, 1e-8, dtype, device)
        
    def forward(self, x, cos_sin):
        x = x + self.attention(self.rmsn_1(x), cos_sin=cos_sin)
        x = x + self.feed_forward(self.rmsn_2(x))
        return x

class Transformer(mn.Module):
    def __init__(self, config: mn.Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.config = config
        self.token_embedding = mn.Embedding(config.vocab_size, config.n_embd, dtype, device)
        #self.pos_embedding = Embedding(config.context_length, config.n_embd, dtype, device)
        self.blocks = mn.Sequential([
            Block(config, dtype, device) for _ in range(config.n_layer)
        ])
        self.rmsn_final = mn.RMSNorm(config.n_embd, 1e-8, dtype, device)
        self.cos, self.sin = mn.precompute_rotary_embeddings(config.n_embd // config.n_head, config.context_length, device=config.device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding(idx) # (B, T) => (B, T, C)
        # REPLACING THIS WITH ROPE
        # pos_embedding = self.pos_embedding(torch.arange(0, T, device = idx.device)) # (T,C)
        x = token_embedding # + pos_embedding # (B, T, C) + (B_b, T, C)
        x = self.blocks(x, cos_sin = (self.cos[:, :, :T, :], self.sin[:, :, :T, :]))
        x = self.rmsn_final(x)
        logits = x @ self.token_embedding.embedding.T # lm_head, but directly applied (B, T, C) @ (C, vocab_size) => (B, T, vocab_size)
        if targets is not None:
            logits_flat = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets, ignore_index=-100)
            return None, loss
        else: 
            loss = torch.zeros((1))
        return logits, loss
    

    """
    exclusively adamw for all optimization. configure_adamw and configure_muon when using them combined.
    """
    def configure_optimizer(self, weight_decay, learning_rate, device):
        params = self.parameters()
        decay_params = [p for p in params if p.ndim >= 2]
        non_decay_params = [p for p in params if p.ndim < 2] # don't decay the biases or 1D vectors
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, eps=1e-8, fused='cuda' in device)
    
    def configure_adamw(self, weight_decay, learning_rate, device):
        params = self.parameters()
        non_decay_params = [p for p in params if p.ndim < 2] # don't decay the biases or 1D vectors
        decay_params = self.token_embedding.parameters()
        optim_groups=[
            {'params': non_decay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': weight_decay}
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, eps=1e-8, fused='cuda' in device)
    
    def configure_muon(self, weight_decay, learning_rate, device):
        params = self.parameters()
        embedding_params = self.token_embedding.parameters()# my .parameters() is not a generator, but a list.
        embd_ids = {id(p) for p in embedding_params}
        hidden_weights = [p for p in params if p.ndim >= 2 and id(p) not in embd_ids]
        
        return torch.optim.Muon(hidden_weights, lr=learning_rate, weight_decay=weight_decay, eps=1e-7, adjust_lr_fn="match_rms_adamw")
    
