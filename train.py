import torch
import myneural as mn

config = mn.Config()
config.n_embd = 192
config.context_length = 256
config.n_head = 6
config.n_layer = 6
config.mini_batch_size = 64
config.device = 'cuda'


text = open('tinyshakespeare.txt', 'r').read()
chars = sorted(list(set(text)))
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
vocab_size = len(chars)
def encode(text):
    return [stoi[s] for s in text]

def decode(list):
    return ''.join([itos[i] for i in list])


model = mn.Transformer(config, dtype=torch.float32, device=config.device)
data = torch.tensor(encode(text), dtype=torch.long, device=config.device)

n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

def get_batch(split):
    x = train if split == 'train' else val
    ix = torch.randint(0, len(x) - config.context_length, (config.mini_batch_size,))
    xs = [x[i:i+config.context_length] for i in ix] # (B, T)
    ys = [x[i+1:i+config.context_length+1] for i in ix] # (B, T)
    return xs, ys

xs, ys = get_batch('train')
xs[:10]
