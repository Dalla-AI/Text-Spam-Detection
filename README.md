# Text-Spam-Detection
Project for classifying SMS messages as spam or not using machine learning. It covers data preparation, model training, and evaluation.

## Features
- ✅ **Easy Data Preparation:** Converts and splits SMS data into train, validation, and test sets.
- ✅ **Baseline Classifier:** Logistic Regression with TF-IDF.
- ✅ **Advanced Models:** Fine-tuned Transformer models (BERT and ELECTRA).
- ✅ **Zero-shot Classification:** Quick experiments with BART models.

## Repository Structure
. ├── data/ │ ├── SMS_Spam.csv # SMS dataset │ └── dataset_splits/ # Train/validation/test sets ├── notebooks/ │ ├── 1_Reading_Parquet_File.py # Data loading script │ ├── 2_Train_Test_Val_Split_Folders.py # Data splitting script │ ├── 3_Model_Training_and_Save.py # Baseline training script │ ├── 4_Fine_Tuning_Bert_Uncased.ipynb # BERT fine-tuning notebook │ ├── 5_Fine_Tuning_Electra.ipynb # ELECTRA fine-tuning notebook │ ├── 6_Zero_Shot_Classification.ipynb # Zero-shot classification notebook │ └── BOW_with_Logistic_Regression.ipynb# Logistic regression notebook ├── models/ │ ├── logistic_regression_model.joblib # Logistic regression model │ └── tfidf_vectorizer.joblib # TF-IDF vectorizer ├── docs/ │ └── HW3_Report.docx # Detailed project report ├── requirements.txt # Required Python packages └── README.md # Project description and guide


## Getting Started

### 🛠️ Installation
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/sms-spam-classification.git
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

📈 Results & Reports
Detailed evaluations and insights are available in docs/HW3_Report.docx and the included notebooks.

🤝 Contributing
Your improvements are welcome! Open an issue or pull request to contribute.

🙏 Acknowledgments

SMS spam dataset (https://huggingface.co/datasets/ucirvine/sms_spam)

Hugging Face Transformers (https://github.com/huggingface/transformers)

Scikit-learn (https://scikit-learn.org/)

