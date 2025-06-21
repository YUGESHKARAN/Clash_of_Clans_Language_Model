import torch
from language_model_class import LanguageModel
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)


with open("tokenizer.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

def encode(s): return [stoi.get(c, stoi[' ']) for c in s]
def decode(l): return ''.join([itos[i] for i in l])

vocab_size = len(stoi)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Hyperparameters
# batch_size = 64
# block_size = 256
# n_embed = 256
# n_heads = 8
# n_layer = 6
# learning_rate = 1e-4
# max_iters = 3000
# eval_interval = 100


# MOdel Architecture

# # Attention class
# class SelfAttentionHead(nn.Module):
#     def __init__(self,head_size):
#         super().__init__()
#         self.query = nn.Linear(n_embed,head_size,bias = False)
#         self.key = nn.Linear(n_embed,head_size,bias = False)
#         self.value = nn.Linear(n_embed,head_size,bias = False)
#         self.dropout = nn.Dropout(0.1)
#         self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

#     def forward(self,x):
#         B,T,C = x.shape
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         wei = q @ k.transpose(-2,-1) * (C**-0.5)
#         wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
#         wei = F.softmax(wei,dim=-1)
#         wei = self.dropout(wei)

#         return wei @ v

# # Transformers class
# class TransformerBlock(nn.Module):
#     def __init__(self,n_embed,n_heads):
#         super().__init__()
#         head_size = n_embed // n_heads
#         self.sa_heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_heads)])
#         self.proj = nn.Linear(n_embed,n_embed)
#         self.ffw = nn.Sequential(
#             nn.Linear(n_embed,n_embed),
#             nn.ReLU(),
#             nn.Linear(n_embed,n_embed)
#         )
#         self.ln1 = nn.LayerNorm(n_embed)
#         self.ln2 = nn.LayerNorm(n_embed)
#         self.dropout = nn.Dropout(0.1)


#     def forward(self,x):
#         x = x + self.dropout(self.proj(torch.cat([h(x) for h in self.sa_heads], dim=-1))) # resnet connection 1
#         x = self.ln1(x)
#         x = x + self.ffw(x) # resnet connection 2
#         x = self.ln2(x)

#         return x
    
# # Language Model class
# class LanguageModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.token_embed = nn.Embedding(vocab_size,n_embed)
#         self.pos_embed = nn.Embedding(block_size,n_embed)
#         self.blocks = nn.Sequential(*[TransformerBlock(n_embed,n_heads) for _ in range(n_layer)])
#         self.ln = nn.LayerNorm(n_embed)
#         self.lm = nn.Linear(n_embed,vocab_size)

#     def forward(self, idx,target=None):

#         B,T = idx.shape
#         static_embed = self.token_embed(idx)
#         positional_embed  = self.pos_embed(torch.arange(T, device=idx.device))
#         x = static_embed + positional_embed
#         x = self.blocks(x)
#         x = self.ln(x)
#         logits = self.lm(x)

#         if target is None:
#             return logits , None

#         B,T,C = logits.shape
#         logits = logits.view(B*T,C)
#         target = target.view(B*T)
#         loss = F.cross_entropy(logits,target)

#         return logits,loss


#     def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
#         for _ in range(max_new_tokens):
#             idx_cond = idx[:, -block_size:]
#             logits, _ = self(idx_cond)
#             logits = logits[:, -1, :] / temperature

#             if top_k is not None:
#                 v, _ = torch.topk(logits, top_k)
#                 logits[logits < v[:, [-1]]] = -float('Inf')

#             probs = F.softmax(logits, dim=-1)
#             next_idx = torch.multinomial(probs, num_samples=1)
#             idx = torch.cat((idx, next_idx), dim=1)

#             # Early stopping if stop_token is generated
#             if stop_token:
#                 stop_token_id = encode(stop_token)[0]
#                 if next_idx.item() == stop_token_id:
#                     break
#         return idx


# MOdel instantiation
model = LanguageModel().to(device)
model.load_state_dict(torch.load("clash_transformer_finetuned.pth"))
model.eval()

def count_parameters_in_millions(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total / 1_000_000,
        "trainable_parameters": trainable / 1_000_000}


@app.route("/")
def index():
    model_info = count_parameters_in_millions(model)
    return jsonify({
        "model_name": "Clash Transformer",
        "total_parameters": model_info["total_parameters"],
        "trainable_parameters": model_info["trainable_parameters"]
    })

# === Flask route ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("prompt", "")
    if not user_input:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = f"<|user|> {user_input} <|assistant|>"
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=400,
            temperature=0.8,
            top_k=40,
            stop_token="<|endoftext|>"
        )

    full_text = decode(output_ids[0].tolist())
    start = full_text.find("<|assistant|>") + len("<|assistant|>")
    end = full_text.find("<|endoftext|>", start)
    assistant_response = full_text[start:end].strip() if end != -1 else full_text[start:].strip()

    return jsonify({
        "prompt": user_input,
        "response": assistant_response
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0' , debug=True)


