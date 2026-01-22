# Data Acquisition Guide

## Dataset Options for AML Fraud Detection

### Option 1: IEEE-CIS Fraud Detection Dataset (Recommended)

**Source:** Kaggle Competition
**URL:** https://www.kaggle.com/c/ieee-fraud-detection/data

**Description:**
- Real-world credit card transactions from Vesta Corporation
- 590,540 transactions for training
- Features include transaction amount, product code, card info, address, email domain, device info
- Binary fraud labels (isFraud)
- Industry-standard dataset used by major financial institutions

**Files to Download:**
- `train_transaction.csv` - Transaction data
- `train_identity.csv` - Identity data
- `test_transaction.csv` - Test data
- `test_identity.csv` - Test identity data

**How to Download:**
1. Create a free Kaggle account at https://www.kaggle.com
2. Go to: https://www.kaggle.com/c/ieee-fraud-detection/data
3. Click "Download All"
4. Extract files to `AML/data/raw/`

---

### Option 2: PaySim Synthetic Financial Dataset

**Source:** Kaggle Datasets
**URL:** https://www.kaggle.com/datasets/ealaxi/paysim1

**Description:**
- Synthetic mobile money transactions dataset
- Simulates mobile money transactions based on real data
- 6.3 million transactions
- Features: step (time), type, amount, origin/destination balances
- Perfect for academic/portfolio projects

**Files to Download:**
- `PS_20174392719_1491204439457_log.csv`

**How to Download:**
1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Click "Download"
3. Extract to `AML/data/raw/`
4. Rename to `paysim_transactions.csv`

---

### Option 3: Credit Card Fraud Detection

**Source:** Kaggle
**URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Description:**
- European cardholder transactions (2013)
- 284,807 transactions
- Highly imbalanced (0.172% fraud)
- Anonymized features (PCA transformed)
- Good for demonstrating imbalanced learning techniques

**Files to Download:**
- `creditcard.csv`

**How to Download:**
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download"
3. Place in `AML/data/raw/`

---

## Installation Instructions

### Step 1: Set Up Python Environment

```bash
# Navigate to project directory
cd AML

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Kaggle API (Optional)

For automated downloads:

```bash
# Install Kaggle API
pip install kaggle

# Set up API credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Place downloaded kaggle.json in:
#    Windows: C:\Users\<YourUsername>\.kaggle\
#    Mac/Linux: ~/.kaggle/

# Download dataset via command line
kaggle competitions download -c ieee-fraud-detection
```

### Step 3: Verify Data Structure

After downloading, your directory should look like:

```
AML/
└── data/
    └── raw/
        ├── train_transaction.csv
        ├── train_identity.csv
        ├── test_transaction.csv
        └── test_identity.csv
```

---

## Data Dictionary

### IEEE-CIS Dataset Key Features

**Transaction Features:**
- `TransactionID`: Unique transaction identifier
- `TransactionDT`: Time delta from reference datetime (seconds)
- `TransactionAmt`: Transaction payment amount (USD)
- `ProductCD`: Product code (W, C, H, S, R)
- `card1-card6`: Payment card information (type, bank, country, etc.)
- `addr1, addr2`: Address information
- `dist1, dist2`: Distance measurements
- `P_emaildomain`: Purchaser email domain
- `R_emaildomain`: Recipient email domain
- `C1-C14`: Counting features (e.g., address match, name match)
- `D1-D15`: Time deltas (e.g., days between previous transaction)
- `M1-M9`: Match features (e.g., name match)
- `V1-V339`: Vesta engineered features

**Identity Features:**
- `id_01-id_11`: Identity information
- `id_12-id_38`: Digital signature features
- `DeviceType`: Device type (mobile/desktop)
- `DeviceInfo`: Device information

**Target:**
- `isFraud`: Binary label (1 = fraud, 0 = legitimate)

---

## Quick Start After Data Download

```python
# Run this in Python to test data loading
import pandas as pd

# Load data
train_txn = pd.read_csv('data/raw/train_transaction.csv')
train_id = pd.read_csv('data/raw/train_identity.csv')

# Merge
df = train_txn.merge(train_id, on='TransactionID', how='left')

print(f"Total transactions: {len(df):,}")
print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")
print(f"Features: {len(df.columns)}")
```

---

## Alternative: Generate Synthetic Data

If you prefer not to download large datasets, you can generate synthetic data:

```python
# See notebooks/00_synthetic_data_generation.ipynb
# This will create a smaller synthetic dataset for testing
```

---

## Data Privacy & Compliance

✅ All recommended datasets are:
- Publicly available
- Anonymized/synthetic
- Safe for portfolio use
- Accepted by recruiters/interviewers

❌ DO NOT use:
- Real customer data from your bank
- Personally identifiable information (PII)
- Proprietary transaction data

---

## Next Steps

After data acquisition:
1. Run `notebooks/01_data_exploration.ipynb` - Explore the data
2. Run `notebooks/02_feature_engineering.ipynb` - Create features
3. Run `notebooks/03_rule_based_aml.ipynb` - Apply AML rules
4. Run `notebooks/04_ml_models.ipynb` - Train ML models
5. Run `notebooks/05_risk_scoring.ipynb` - Generate risk scores

---

## Troubleshooting

**Issue:** "File not found"
- Verify files are in `data/raw/` directory
- Check file names match exactly

**Issue:** "Memory error"
- Use data sampling: `df = pd.read_csv('file.csv', nrows=100000)`
- Reduce feature count
- Use chunking for large files

**Issue:** "Import error"
- Verify virtual environment is activated
- Re-run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

---

For questions or issues, refer to the main README.md
