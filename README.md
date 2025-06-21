# Clash of Clans Language Model 🏰🧠
--> A mini language model developed from scratch using PyTorch <br>
--> fine-tuned on a supervised Clash of Clans dataset (clash_finetune_chat_100.jsonl).<br>
--> The model has approximately **2.96 million** total and trainable parameters.<br>

### Prerequisites

- 🐍 Python 3.8+
- 📓 Jupyter Notebook
- 💡 Recommended: virtualenv or conda

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YUGESHKARAN/Clash_of_Clans_Language_Model.git
    cd Clash_of_Clans_Language_Model
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    # Or, for conda users:
    # conda env create -f environment.yml
    ```

# 📁 Project Structure

Clash_of_Clans_Language_Model/ <br>
├── clash_finetune_chat_100.jsonl   # supervised dataset with c.o.c heros and toops info. <br>
├── clash_of_clans.ipynb            # Jupyter Notebooks for exploration and experiments <br>
├── index. py                       # Saved or pre-trained models <br>
├── requirements.txt                # Python dependencies <br>
├── environment.yml                 # (Optional) Conda  <br>
└── README.md                       # Project documentation <br>

## Hyperparameters <br>
batch_size = 64 <br>
block_size = 256 <br>
n_embed = 256 <br>
n_heads = 8 <br>
n_layer = 6 <br>
learning_rate = 1e-4 <br>
max_iters = 3000 <br>
eval_interval = 100 <br>

## 🤝 Contributing

Contributions are welcome! Open issues or pull requests for suggestions, bug fixes, or improvements.

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

##  Acknowledgements

- Clash of Clans (Supercell) for inspiration
- Open-source ML and NLP communities
