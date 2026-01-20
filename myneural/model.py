import torch
import myneural as mn
from sentencepiece import SentencePieceProcessor as sp
class Block(mn.Module):
    def __init__(self, config: mn.Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.attention = mn.MultiHeadAttention(config, dtype, device)
        self.feed_forward = mn.FeedForward(config, dtype, device)
        self.rmsn_1 = mn.RMSNorm(config.n_embd, 1e-8, dtype, device)
        self.rmsn_2 = mn.RMSNorm(config.n_embd, 1e-8, dtype, device)
        
    def forward(self, x):
        x = x + self.attention(self.rmsn_1(x))
        x = x + self.feed_forward(self.rmsn_2(x))
        return x

class Transformer(mn.Module):
    def __init__(self, config: mn.Config, dtype = torch.float32, device = 'cpu'):
        super().__init__()
        self.config = config
        self.token_embedding = mn.Embedding(config.vocab_size, config.n_embd, dtype, device)
        self.pos_embedding = mn.Embedding(config.context_length, config.n_embd, dtype, device)
        self.transformer = mn.Sequential([
            Block(config, dtype, device) for _ in range(config.n_layer)
        ])
        self.rmsn_final = mn.RMSNorm(config.n_embd, 1e-8, dtype, device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding(idx) # (B, T) => (B, T, C)
        pos_embedding = self.pos_embedding(torch.arange(0, T, device = idx.device)) # (T,C)
        x = token_embedding + pos_embedding # (B, T, C) + (B_b, T, C)
        x = self.transformer(x)
        x = self.rmsn_final(x)
        logits = x @ self.token_embedding.embedding.T # lm_head, but directly applied (B, T, C) @ (C, vocab_size) => (B, T, vocab_size)
        if targets is not None:
            logits_flat = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets)
        else: 
            loss = torch.zeros((1))
        return logits, loss
    


    def configure_optimizer(self, weight_decay, learning_rate, device):
        params = self.parameters()
        decay_params = [p for p in params if p.ndim >= 2]
        non_decay_params = [p for p in params if p.ndim < 2] # don't decay the biases or 1D vectors
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, eps=1e-8, fused='cuda' in device)
 
@torch.no_grad()
def generate(model, idx, max_new_tokens):
    for i in range(max_new_tokens):
        idx_cut = idx[:, -model.config.context_length:]
        logits, _ = model(idx_cut) # (B, T, C)
        logits = logits[:, -1, :] # last token, (B, C)
        probs = torch.softmax(logits, dim = -1)
        idx_new = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_new], dim=-1) 
    return idx 
 
@torch.no_grad()   
def prompt(prompt: str, model, tokenizer: sp, new_tokens = 256, debug=False, device = 'cpu'):
    model.eval()
    tokens = generate(model, torch.tensor([[tokenizer.eos_id()] + tokenizer.Encode(prompt)], dtype=torch.long, device=device), new_tokens) # (B, T), but B = 1
    tokens = tokens[0].tolist() # (T,), converted from tensor to pylist
    model.train()
    return ''.join(tokenizer.IdToPiece(id) for id in tokens).replace('▁', ' ').replace('</s>', '</s>\n') if debug else tokenizer.Decode(tokens)

    