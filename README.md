# claims-fraud-detection

An end-to-end workflow for detecting potential fraud in auto/insurance claims: Exploratory Data Analysis (EDA), optional ingestion into SQLite, classification modeling (including XGBoost), and optional LLM-based business explanations using LangChain + Ollama. 

## Notebooks
| Notebook | What it does |
|---|---|
| `EDA.ipynb` | EDA, missing-value handling for several categorical columns, timestamp feature engineering from incident date, and scaling preparation. |
| `CSV-to-DB.ipynb` | Loads the dataset from CSV and writes it into a SQLite database (`claims.db`) as table `claims`, then inspects the table schema. |
| `Classification-Modelling.ipynb` | Fraud classification pipeline using preprocessing (OneHotEncoder), train/test split, feature selection via RFECV, and model comparison (LR/KNN/DT/RF/XGBoost). Includes evaluation (classification report, confusion matrix) and threshold check. |
| `LangChain.ipynb` | Uses LangChain with Ollama (e.g., `llama2`) to generate business-friendly explanations for why a claim could be fraudulent, based on selected features. |

## Dataset
Place your dataset at:
- `dataset/claims.csv`
- source: AQQAD, ABDELRAHIM (2023), “insurance_claims ”, Mendeley Data, V2, doi: 10.17632/992mh7dk9y.2

The target label used in modeling is `fraudreported` and is mapped to a binary label (e.g., Y/N to 1/0).

## Recommended run order
1. `EDA.ipynb`
2. (Optional) `CSV-to-DB.ipynb`
3. `Classification-Modelling.ipynb`
4. `LangChain.ipynb`

## Environment (suggested)
- Python 3.x
- Core libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- Modeling: xgboost
- DB: sqlite3 (built-in)
- LLM: langchain-core, langchain-ollama, plus Ollama installed locally

## Outputs
- EDA insights and cleaned/engineered features
- `claims.db` SQLite database (optional)
- Trained/evaluated ML models (best model indicated in the modeling notebook)
- LLM-generated explanations for flagged claims
