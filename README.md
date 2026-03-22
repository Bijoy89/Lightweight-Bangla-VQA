# Lightweight Bangla VQA

Lightweight Visual Question Answering for Bangla using EfficientNet-B0 + BiLSTM.

This project implements a CNN–LSTM–based VQA system designed for
low-resource languages with small model size and fast inference.

The repository contains:

- Full training notebook
- Model code
- Inference pipeline
- Streamlit app
- Vocabulary files

The trained model file is NOT included because it exceeds GitHub's size limit.
Users can reproduce the model by running the training notebook.


# Features

- EfficientNet-B0 visual encoder
- BiLSTM question encoder
- Question-guided spatial attention
- Concat fusion
- Autoregressive LSTM decoder
- Beam search decoding
- 16M parameters only
- Runs on CPU / GPU



# Results

Test set (Bangla-Bayanno)

| Metric | Score |
|--------|--------|
| Char BLEU-1 | 0.44 |
| Char BLEU-4 | 0.30 |
| Exact Match | 37% |
| Token Accuracy | 33% |
| F1 | 0.38 |

# Model Size

Total params: 16.05M

| Part | Params |
|------|--------|
| EfficientNet-B0 | 4M |
| VQA model | 12M |

Inference ≈ 13ms per sample (T4 GPU)


# Architecture

```
Image → EfficientNet → Feature map
Question → Embedding → BiLSTM
Attention → Image context
Concat fusion
LSTM decoder → Answer
```

---

# Dataset

Primary dataset: Bangla-Bayanno

48k QA pairs after filtering

Split:

Train 70%  
Val 15%  
Test 15%

Vocabulary:

Question vocab 4750  
Answer vocab 1747  

Most answers are single word.

---

## Installation

Clone repo


git clone https://github.com/Bijoy89/Lightweight-Bangla-VQA.git
cd Lightweight-Bangla-VQA


Create env

Windows

```
python -m venv venv
venv\Scripts\activate
```

Linux / Mac

```
python -m venv venv
source venv/bin/activate
```

Install packages

```
pip install -r requirements.txt
```

---

## Training the Model

The trained model is not included.

To generate best_model.pt run the whole training notebook.

Open the notebook:

```
banglavqafinal2.ipynb
```

Recommended:

Run on Kaggle GPU (T4)

After training finishes, you will get:

```
best_model.pt
q_stoi.json
a_stoi.json
```

Create folder:

```
checkpoints
```

Put model:

```
checkpoints/best_model.pt
```

Make sure vocab files match the checkpoint.

---

## Run App

```
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

## How to Use

1 Upload image  
2 Type Bangla question  
3 Click Get Answer  

Works best on:

- yes/no
- color
- counting
- sports
- room type

Weak on:

- long answers
- unseen words
- brand names
- text reading

## Training Details

Trained on Kaggle T4 GPU

Stages:

Stage1 ablation  
Stage2 training  
Extended training  

Total ≈ 40 epochs

---

# Limitations

- Exact match ≈ 37%
- single-word bias
- small vocab
- CNN-LSTM, not transformer
- cross-dataset drop

This is a research prototype.

---

## License

MIT
