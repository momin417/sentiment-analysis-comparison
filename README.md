# Sentiment Analysis: TF-IDF Baseline vs. Fine-Tuned DistilBERT

A reproducible sentiment classification pipeline that compares a classical machine learning baseline (TF-IDF + Logistic Regression / Linear SVC / Naive Bayes) against a fine-tuned DistilBERT transformer. Built as a self-contained Google Colab notebook with automatic column detection, full EDA, preprocessing, training, evaluation, error analysis, and a reusable inference function.

## Features

- **Automatic column detection** — infers text and label columns from any CSV, or lets you set them manually
- **Exploratory Data Analysis** — class balance, missing values, duplicates, text length distribution, sample inspection
- **Text preprocessing** — HTML unescaping, URL/tag stripping, negation-aware stopword removal, lemmatization
- **Flexible label normalization** — supports categorical labels or numeric ratings mapped to negative/neutral/positive via configurable thresholds
- **Two model families compared on the same held-out test set**
  - Classical: TF-IDF (unigrams + bigrams) with Logistic Regression, Linear SVC, and Multinomial Naive Bayes
  - Deep learning: fine-tuned `distilbert-base-uncased` via Hugging Face `Trainer`
- **Class imbalance handling** via `class_weight='balanced'` for classical models
- **Full evaluation** — accuracy, macro precision/recall/F1, classification reports, confusion matrices
- **Error analysis** — surfaces misclassified examples from the best-performing model for qualitative review
- **Reusable inference function** — `predict_sentiment(text, model_choice)` for ad-hoc predictions with either model

## Tech Stack

- Python, pandas, NumPy, scikit-learn
- Hugging Face `transformers`, `datasets`, `evaluate`
- PyTorch
- NLTK (tokenization, stopwords, lemmatization)
- matplotlib, seaborn

## Getting Started

### Option 1: Google Colab (recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run the setup cell to install dependencies.
3. When prompted, upload `train.csv` and `test.csv` (or mount Google Drive and update the file paths in the config cell).
4. Run all cells in order.

A GPU runtime (`Runtime > Change runtime type > GPU`) is strongly recommended for the transformer fine-tuning step.

### Option 2: Local / Jupyter

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

Place `train.csv` and `test.csv` in the project root, or update `TRAIN_PATH` / `TEST_PATH` in the configuration cell. Remove or skip the `google.colab` upload cell when running locally.

## Data Format

The notebook expects two CSV files:

- `train.csv` — used for EDA, training, and validation (auto-split 85/15)
- `test.csv` — held out and only used for final evaluation

Each file needs at minimum one text column and one label column. Column names are auto-detected but can be overridden via `TEXT_COLUMN` and `LABEL_COLUMN` in the config cell. Numeric rating labels (e.g. 1–5 stars) are automatically bucketed into negative / neutral / positive using `NEGATIVE_MAX` and `POSITIVE_MIN`.

> **Note:** No dataset is included in this repository. Bring your own `train.csv` / `test.csv`, or point the paths at a public dataset (e.g. Sentiment140, IMDB reviews, Twitter sentiment datasets).

## Project Structure

```
.
├── sentiment_analysis.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
├── requirements.txt           # Python dependencies
└── README.md
```

## Results

The notebook prints a side-by-side comparison table (accuracy, macro precision/recall/F1, training time) and a bar chart comparing macro F1 for the baseline vs. the transformer at the end of the run. Since results depend on the dataset you supply, they aren't hardcoded here — re-run the notebook on your data to generate your own comparison table and confusion matrices.

## Configuration

Key parameters (set at the top of the notebook):

| Variable | Purpose |
|---|---|
| `TEXT_COLUMN` / `LABEL_COLUMN` | Override auto-detected column names |
| `MAX_SAMPLES` | Cap on training rows used for transformer fine-tuning (for runtime control) |
| `NEGATIVE_MAX` / `POSITIVE_MIN` | Thresholds for bucketing numeric labels into negative/neutral/positive |
| `TRANSFORMER_MODEL_NAME` | Hugging Face model checkpoint (default: `distilbert-base-uncased`) |
| `TRANSFORMER_EPOCHS` / `TRANSFORMER_BATCH_SIZE` | Auto-adjusted based on GPU availability |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
