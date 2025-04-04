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

Data Preparation

If needed, convert SMS data (in Parquet format) to CSV format with:

Use notebooks/1_Reading_Parquet_File.py (or the equivalent notebook/script) to load and convert your data if needed.

Dataset Splitting

Run notebooks/2_Train_Test_Val_Split_Folders.py to split the data into training, validation, and test sets.

Model Training

Baseline (Logistic Regression):
Run notebooks/3_Model_Training_and_Save.py or open notebooks/BOW_with_Logistic_Regression.ipynb to train and evaluate the baseline model.

Fine-tuned Transformers:
Open notebooks/4_Fine_Tuning_Bert_Uncased.ipynb or notebooks/5_Fine_Tuning_Electra.ipynb to fine-tune the respective transformer models.

Zero-shot Classification

Open and run notebooks/6_Zero_Shot_Classification.ipynb to experiment with zero-shot classification using BART models.

BOW (Bag of Words) Baseline using TF-IDF and Logistic Regression
Open and run notebooks/7 BOW with Logistic Regression.py to compare all other tested models to simple logistic regression model.

Model Artifacts

The trained models and vectorizers are saved in the models/ folder (e.g., logistic_regression_model.joblib, tfidf_vectorizer.joblib).



🤝 Contributing
Your improvements are welcome! Open an issue or pull request to contribute.

🙏 Acknowledgments

SMS spam dataset (https://huggingface.co/datasets/ucirvine/sms_spam)

Hugging Face Transformers (https://github.com/huggingface/transformers)

Scikit-learn (https://scikit-learn.org/)

