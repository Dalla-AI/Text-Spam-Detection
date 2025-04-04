# Text-Spam-Detection
Project for classifying SMS messages as spam or not using machine learning. It covers data preparation, model training, and evaluation.

## Features
- ✅ **Data Preparation:** Converts and splits SMS data into train, validation, and test sets.
- ✅ **Baseline Classifier:** Logistic Regression with TF-IDF.
- ✅ **Fine-Tuning Models:** Fine-tuned Transformer models (BERT and ELECTRA).
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

## 🚀 Usage

Data Preparation

If needed, convert SMS data (in Parquet format) to CSV format with:

1._Reading_Parquet_File.py (or the equivalent notebook/script) to load and convert your data.

Dataset Splitting

Run 2._Train_Test_Val_Split_Folders.py to split the data into training, validation, and test sets.

Model Training

Baseline (Logistic Regression)

Run 3._Model_Training_and_Save.py to train and save logistic regression model and vectorizer.

Fine-tuned Transformers

Run 4._Fine_Tuning_Bert_Uncased.ipynb or 5._Fine_Tuning_Electra.ipynb to fine-tune the respective transformer models.

Zero-shot Classification

Run 6._Zero_Shot_Classification.ipynb to experiment with zero-shot classification using BART models.

BOW (Bag of Words) Baseline using TF-IDF and Logistic Regression
Similar to 3, Run 7. BOW with Logistic Regression.py to compare all other tested models to simple logistic regression model.


The trained models and vectorizers are saved in the models/ folder (e.g., logistic_regression_model.joblib, tfidf_vectorizer.joblib).


## 🤝 Contributing
Your improvements are welcome! Message or open an issue/pull request to contribute.

## 🙏 Acknowledgments

SMS spam dataset (https://huggingface.co/datasets/ucirvine/sms_spam)

Hugging Face Transformers (https://github.com/huggingface/transformers)

Scikit-learn (https://scikit-learn.org/)

