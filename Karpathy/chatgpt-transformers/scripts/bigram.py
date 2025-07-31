# a single consolidated script containing the bigram model 

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

# super simple bigram model - introduce position embeddings too but not used in this bigram class. 
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # all lookup tables 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm = nn.Linear(n_embd, vocab_size) # since logits are prob over vocab_size need to transform

    def forward(self, idx, targets=None):
        
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd) -- nn.Embedding has a __call__ method
        # overall encoding includes identity AND position
        x = tok_emb + pos_emb # (B,T,n_embd)
        logits = self.lm(x) # (B,T,vocab_size)
  
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# --------------------------------------RESULTS----------------------------------------------------

# step 0: train loss 4.7305, val loss 4.7241
# step 300: train loss 2.8110, val loss 2.8249
# step 600: train loss 2.5434, val loss 2.5682
# step 900: train loss 2.4932, val loss 2.5088
# step 1200: train loss 2.4863, val loss 2.5035
# step 1500: train loss 2.4665, val loss 2.4921
# step 1800: train loss 2.4683, val loss 2.4936
# step 2100: train loss 2.4696, val loss 2.4846
# step 2400: train loss 2.4638, val loss 2.4879
# step 2700: train loss 2.4738, val loss 2.4911

# od nos CAy go ghanoray t, co haringoudrou clethe k,LARof fr werar,
# Is fa!


# Thilemel cia h hmboomyorarifrcitheviPO, tle dst f qur'dig t cof boddo y t o ar pileas h mo wierl t,
# S:
# STENENEat I athe thounomy tinrent distesisanimald 3I: eliento ald, avaviconofrisist me Busarend un'soto vat s k,
# SBRI he the f wendleindd t acoe ts ansu, thy ppr h.QULY:
# KIIsqu pr odEd ch,
# APrnes ouse bll owhored miner t ooon'stoume bupromo! fifoveghind hiarnge s.
# MI aswimy or m, wardd tw'To tee abifewoetsphin sed The a
