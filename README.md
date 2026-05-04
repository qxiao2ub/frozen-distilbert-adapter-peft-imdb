# Frozen DistilBERT Adapter PEFT IMDB Classifier

**Recommended GitHub repo name:** `frozen-distilbert-adapter-peft-imdb`

**Short description:** Compare full fine-tuning, LoRA, and a frozen DistilBERT bottleneck-adapter classifier on IMDB sentiment classification.

## GitHub topics / keywords

`distilbert`, `adapter-tuning`, `peft`, `imdb`, `sentiment-classification`, `frozen-encoder`, `bottleneck-adapter`, `transformers`, `pytorch`, `model-comparison`

## Project overview

This repository contains a Colab-ready notebook for evaluating three DistilBERT-based training strategies on the IMDB movie-review sentiment classification benchmark:

1. **Full fine-tuning** - all DistilBERT parameters and the classifier are trained.
2. **LoRA PEFT** - LoRA modules are inserted into selected attention projections.
3. **Frozen DistilBERT + bottleneck adapter** - the DistilBERT encoder is frozen, and only a lightweight residual bottleneck adapter plus classifier head are trained.

The project highlights the tradeoff between predictive performance and the number of trainable parameters.

## Source notebook

- `frozen-distilbert-adapter-peft-imdb.ipynb`

## Dataset

The notebook uses the Hugging Face `imdb` dataset:

- Task: binary sentiment classification
- Labels: negative / positive
- Training/validation pool: 10,000 examples sampled from `imdb["train"]`
- Training examples: 8,000
- Validation examples: 2,000
- Test examples: full 25,000-example `imdb["test"]` split
- Maximum sequence length: 256

## Model designs

### 1. Full fine-tuning

A standard `AutoModelForSequenceClassification` model is initialized from `distilbert-base-uncased`. All model parameters are trainable.

### 2. LoRA PEFT

LoRA modules are applied to the DistilBERT attention projection layers:

- Target modules: `q_lin`, `v_lin`
- Rank: 8
- Alpha: 16
- Dropout: 0.10
- Trainable components: LoRA parameters and classification head

### 3. Frozen DistilBERT bottleneck adapter

The adapter classifier uses `AutoModel` as a frozen DistilBERT encoder, extracts the first-token hidden representation, and applies a residual bottleneck adapter before classification.

Adapter formula:

```text
h' = h + W_up ReLU(W_down h)
```

Implementation details:

- Encoder: frozen `distilbert-base-uncased`
- Bottleneck dimension: 64
- Adapter dropout: 0.10
- Classifier dropout: 0.20
- Trainable components: bottleneck adapter and classifier head
- Initialization: near-identity residual initialization

## Hyperparameters

| Model | Trainable components | Learning rate | Epochs | Batch size | Weight decay | Extra settings |
|---|---|---:|---:|---:|---:|---|
| Full FT | All DistilBERT + classifier | 2e-5 | 2 | 16 | 0.01 | None |
| LoRA | LoRA q/v projections + classifier | 1e-4 | 2 | 16 | 0.01 | r=8, alpha=16, dropout=0.10 |
| Adapter | Bottleneck adapter + classifier | 1e-3 | 2 | 16 | 0.01 | bottleneck=64, ReLU, dropout=0.10 |

Other shared settings:

- Eval batch size: 64
- Gradient clipping: 1.0
- Warmup ratio: 0.06
- Maximum sequence length: 256

## Test-set results

| Model | Trainable parameters | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | 66,955,010 | 0.8992 | 0.8882 | 0.9133 | 0.9006 |
| LoRA | 739,586 | 0.8797 | 0.8776 | 0.8825 | 0.8801 |
| Adapter | 100,674 | 0.8410 | 0.8330 | 0.8529 | 0.8428 |

## Key finding

Full fine-tuning gives the strongest predictive performance, LoRA provides a strong parameter-efficient compromise, and the frozen DistilBERT bottleneck adapter uses the fewest trainable parameters. The adapter trains only about **0.15 percent** as many parameters as full fine-tuning, but with a noticeable decrease in test-set F1.

## Repository structure

```text
.
├── Answer_for_problem2.ipynb
├── README.md
└── requirements.txt
```

A minimal `requirements.txt` can contain:

```text
datasets
transformers
peft
accelerate
scikit-learn
pandas
matplotlib
tqdm
torchao
```

## How to run

### Option 1: Google Colab

1. Upload `Answer_for_problem2.ipynb` to Google Colab.
2. Select a GPU runtime.
3. Run the installation cell if needed.
4. Execute the notebook from top to bottom.

### Option 2: Local environment

```bash
git clone https://github.com/<your-username>/frozen-distilbert-adapter-peft-imdb.git
cd frozen-distilbert-adapter-peft-imdb

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install -r requirements.txt
jupyter notebook
```

Then open `Answer_for_problem2.ipynb` and run all cells.

## Main outputs

The notebook produces:

- Shared IMDB train/validation/test splits
- Shared tokenizer and dataloaders
- Training loops for full fine-tuning, LoRA, and adapter-based PEFT
- Loss curves for each method
- Parameter-count reports
- Final test-set comparison table
- Accuracy, precision, recall, and F1 metrics

## Notes

- This adapter implementation applies the bottleneck adapter after the frozen encoder output, using the first-token hidden representation.
- It is not a layer-wise adapter inserted after every Transformer block.
- Results can vary slightly depending on random seed, GPU type, package versions, and runtime environment.
- The Hugging Face dataset and pretrained model are downloaded at runtime.

## Suggested future improvements

- Implement layer-wise adapters inside each Transformer block.
- Add runtime and memory profiling.
- Compare adapter bottleneck sizes such as 16, 32, 64, and 128.
- Add calibration metrics and confusion matrices.
- Export the best PEFT model for lightweight inference.

## License

