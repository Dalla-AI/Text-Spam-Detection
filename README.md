# Text-Spam-Detection
Project for classifying SMS messages as spam or not using machine learning. It covers data preparation, model training, and evaluation.

## Features
- ✅ **Easy Data Preparation:** Converts and splits SMS data into train, validation, and test sets.
- ✅ **Baseline Classifier:** Logistic Regression with TF-IDF.
- ✅ **Advanced Models:** Fine-tuned Transformer models (BERT and ELECTRA).
- ✅ **Zero-shot Classification:** Quick experiments with BART models.

## Getting Started

### 🛠️ Installation
1. **Clone the repository:**
```bash
git clone https://github.com/dalla-AI/sms-spam-classification.git
cd sms-spam-classification
```
```bash
pip install -r requirements.txt
```

🚀 Usage
Prepare Data: Run scripts from notebooks/ to prepare and split your data.

Baseline Model: Use Logistic Regression notebooks/scripts for a quick baseline.

Advanced Models: Fine-tune BERT or ELECTRA using provided notebooks.

Zero-shot Classification: Experiment directly with BART models without additional training.

🤝 Contributing
Your improvements are welcome! Open an issue or pull request to contribute.

🙏 Acknowledgments

SMS spam dataset (https://huggingface.co/datasets/ucirvine/sms_spam)

Hugging Face Transformers (https://github.com/huggingface/transformers)

Scikit-learn (https://scikit-learn.org/)

