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


