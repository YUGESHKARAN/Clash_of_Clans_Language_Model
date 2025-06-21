import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# Hyperparameters
batch_size = 64
block_size = 256
n_embed = 256
n_heads = 8
n_layer = 6
learning_rate = 1e-4
max_iters = 3000
eval_interval = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("tokenizer.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

def encode(s): return [stoi.get(c, stoi[' ']) for c in s]
def decode(l): return ''.join([itos[i] for i in l])

vocab_size = len(stoi)

# MOdel Architecture

# Attention class
class SelfAttentionHead(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embed,head_size,bias = False)
        self.key = nn.Linear(n_embed,head_size,bias = False)
        self.value = nn.Linear(n_embed,head_size,bias = False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        return wei @ v

# Transformers class
class TransformerBlock(nn.Module):
    def __init__(self,n_embed,n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.ffw = nn.Sequential(
            nn.Linear(n_embed,n_embed),
            nn.ReLU(),
            nn.Linear(n_embed,n_embed)
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(0.1)


    def forward(self,x):
        x = x + self.dropout(self.proj(torch.cat([h(x) for h in self.sa_heads], dim=-1))) # resnet connection 1
        x = self.ln1(x)
        x = x + self.ffw(x) # resnet connection 2
        x = self.ln2(x)

        return x
    
# Language Model class
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size,n_embed)
        self.pos_embed = nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed,n_heads) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm = nn.Linear(n_embed,vocab_size)

    def forward(self, idx,target=None):

        B,T = idx.shape
        static_embed = self.token_embed(idx)
        positional_embed  = self.pos_embed(torch.arange(T, device=idx.device))
        x = static_embed + positional_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm(x)

        if target is None:
            return logits , None

        B,T,C = logits.shape
        logits = logits.view(B*T,C)
        target = target.view(B*T)
        loss = F.cross_entropy(logits,target)

        return logits,loss


    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

            # Early stopping if stop_token is generated
            if stop_token:
                stop_token_id = encode(stop_token)[0]
                if next_idx.item() == stop_token_id:
                    break
        return idx
