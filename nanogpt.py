import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters (These hyperparameters can be used if GPU is available)
# batch_size = 64 # how many independent sequences will we process in parallel?
# block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# ------------

# hyperparameters
batch_size = 8  # reduced batch size
block_size = 128  # reduced context length
max_iters = 2000  # fewer iterations
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100  # fewer evaluation iterations
n_embd = 128  # smaller embedding size
n_head = 4  # fewer attention heads
n_layer = 4  # fewer transformer layers
dropout = 0.1  # reduced dropout

torch.manual_seed(1337)

# Read input text
with open('.\\data\\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary and mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # char to index
itos = {i: ch for i, ch in enumerate(chars)}  # index to char
encode = lambda s: [stoi[c] for c in s]  # encoder: string to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: list of integers to string

# Train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% train, 10% validation
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Function to estimate loss on train and val sets
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Self-attention head module
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # mask out upper triangle
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

# Multi-head attention module
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate heads
        out = self.dropout(self.proj(out))
        return out

# Feedforward module
class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Transformer block module
class Block(nn.Module):
    """Transformer block: self-attention followed by feedforward."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # add residual connection with layer normalization
        x = x + self.ffwd(self.ln2(x))  # add residual connection with layer normalization
        return x

# GPT-style language model
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)  # initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Transformer blocks
        x = self.blocks(x)  # (B, T, n_embd)

        # Final layer normalization and linear projection
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is not None:
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generate new tokens based on input context idx
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop to last block_size tokens
            logits, _ = self(idx_cond)  # get logits for next token
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to running sequence
        return idx

# Instantiate model and move to GPU if available
model = GPTLanguageModel()
model.to(device)

# Print number of parameters in the model
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
open('.\\data\\output.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))