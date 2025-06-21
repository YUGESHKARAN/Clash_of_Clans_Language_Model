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

Clash_of_Clans_Language_Model/
├── clash_finetune_chat_100.jsonl   # supervised dataset with c.o.c heros and toops info.
├── clash_of_clans.ipynb            # Jupyter Notebooks for exploration and experiments
├── index. py                       # Saved or pre-trained models
├── requirements.txt                # Python dependencies
├── environment.yml                 # (Optional) Conda environment
└── README.md                       # Project documentation

## Hyperparameters
batch_size = 64
block_size = 256
n_embed = 256
n_heads = 8
n_layer = 6
learning_rate = 1e-4
max_iters = 3000
eval_interval = 100

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for suggestions, bug fixes, or improvements.

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

##  Acknowledgements

- Clash of Clans (Supercell) for inspiration
- Open-source ML and NLP communities
