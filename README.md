# CFPB Complaint Classification

Machine Learning course project. It automatically routes consumer complaints to the right department (credit card, mortgage, debt collection, etc.) instead of sorting them by hand.

## Data

Source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) (US government).
Download: (https://www.kaggle.com/datasets/selener/consumer-complaint-database)

The data files are **not** included in this repo because they are too large.
Download the CSV, rename it to `rows.csv`, and put it in a folder named `data/`.

## Project Structure

```
ticket_project/
├── data/                    # not in repo (rows.csv, clean.csv)
├── preprocessing.py         # cleans the data and saves data/clean.csv
├── logistic_regression.py   # classifies complaints into 9 departments
├── linear_regression.py     # predicts days until the complaint is sent to the company
├── project.ipynb            # data exploration notebook
└── README.md
```

## How to Run

1. Install the requirements:
   ```
   pip install pandas numpy scikit-learn matplotlib scipy
   ```
2. Put `rows.csv` inside `data/`.
3. Run the files in this order:
   ```
   python preprocessing.py          # creates data/clean.csv (~1.2M rows -> ~380K rows)
   python logistic_regression.py    # reads data/clean.csv
   python linear_regression.py      # independent, reads data/rows.csv
   ```

## Steps

1. **Preprocessing:** drops empty rows, merges old/new category names, cleans the text, encodes labels, saves `clean.csv`.
2. **Logistic Regression (main model):** TF-IDF (5000 features) + Logistic Regression over 9 classes.
   - Accuracy: 0.839
   - Macro AUC: 0.974
   - Weakest classes: Payday loan and Vehicle loan (smaller classes).
3. **Linear Regression (side experiment):** predicts the number of days between "Date received" and "Date sent to company".
   - R2: 0.034 (weak). Most complaints are sent on the same day (0 days), so the text does not explain the delay.
#
