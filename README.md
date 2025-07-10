# TF-BERT Word Imputer

This project implements a **Masked Language Model (MLM)** using `TFBertForMaskedLM` to predict missing words in sentences. It utilizes the **BERT-Large Uncased** model fine-tuned on a custom sentence dataset with dynamic token masking. The final model achieves a validation **perplexity of 8.48** and Test-Set **perplexity of 10.51**.

### Model architecture:
![Model Architecture](assets/architecture.png)

### Learning rate schedule:
![lr_schedule](assets/lr_schedule.png)
---

## 🧰 Features

* Built using **TensorFlow** and **Hugging Face Transformers** with `TFBertForMaskedLM`.
* Custom dataset loaded via Hugging Face `datasets` library.
* Dynamic masking using `DataCollatorForLanguageModeling`.
* Polynomial learning rate decay and Adam optimizer.
* Early stopping for stability.
* Final model saved in `assets/mlm_tf` directory.

---

## 📊 Training Configuration

```python
model_name = "bert-large-uncased"
max_seq_length = 128
batch_size = 8
num_train_epochs = 10
learning_rate = 5e-6 (highest)
mlm_probability = 0.15
```

---
## 🔹 Results

* **Test Perplexity**: 10.51
* **Model Size**: \~1.3GB (BERT-Large)

---

## 💼 Use Case

This model can be used for:

* Filling in blanks in educational apps
* Grammatical sentence correction
* Augmenting incomplete text data