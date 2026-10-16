# Rainfall Prediction Using LightGBM (Light Gradient Boosting Machine)

Predicting **whether it will rain tomorrow** from today's weather observations, using gradient boosting (LightGBM, XGBoost), classic machine learning models, a voting ensemble, and deep learning models (GRU-based RNN and 1D-CNN).

---

## Motivation

Accurate rainfall prediction supports decision-making in **agriculture, water resource management, disaster management and urban planning**. This matters especially in agricultural countries like India, where rainfall affects the economy and the livelihoods of millions.

LightGBM is the main model because it handles large datasets efficiently and reduces training time through histogram-based, leaf-wise tree splitting. Its performance is compared with XGBoost, Random Forest, K-Nearest Neighbors, Gaussian Naive Bayes, a hard-voting ensemble, and two neural networks.

---

## Dataset

The project uses the **Rain in Australia** dataset from Kaggle: about 10 years of daily weather observations from weather stations across Australia.

| Property | Value |
|----------|-------|
| File | `DatasetAUS.csv` |
| Records | 142,193 daily observations (with a known target) |
| Columns | 23 |
| Target | `RainTomorrow` (Yes / No) |
| Class balance (test set) | ~80% No rain, ~20% Rain |

**Features**

| Type | Columns |
|------|---------|
| Temperature | `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm` |
| Rain and sun | `Rainfall`, `Evaporation`, `Sunshine`, `RainToday` |
| Wind | `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm` |
| Humidity and pressure | `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm` |
| Cloud cover | `Cloud9am`, `Cloud3pm` |
| Other | `Date`, `Location` |

---

## Methodology

```
DatasetAUS.csv
      │
      ▼
1. Preprocessing ── encode target, fill missing values,
      │              encode categoricals, correlation-based feature selection,
      │              z-score standardization
      ▼
2. Train/test split (80 / 20)
      │
      ├──► Gradient boosting: LightGBM, XGBoost
      ├──► Classic ML: Random Forest, KNN (k = 7), Gaussian Naive Bayes
      ├──► Hard-voting ensemble of all five
      └──► Deep learning: GRU (RNN) and 1D-CNN on 30-day windows
      │
      ▼
3. Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
```

### 1. Data Preprocessing
- **Target encoding:** `RainTomorrow` converted to 0/1 with `pd.factorize`.
- **Missing values:** numeric columns filled with the column mean; the remaining gaps filled forward. `Sunshine` (~48%), `Evaporation` (~43%) and the cloud columns (~38–40%) have the most missing data.
- **Categorical encoding:** `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` and `Date` label-encoded.
- **Feature selection:** features with an absolute correlation below 0.05 with the target were dropped (for example `Location` and `Temp9am`).
- **Standardization:** numeric features scaled to zero mean and unit variance.

**Strongest predictors** (correlation with `RainTomorrow`): `Humidity3pm` (+0.44), `Sunshine` (−0.33), `Cloud3pm` (+0.30), `Humidity9am` (+0.26), `Cloud9am` (+0.25), `Rainfall` (+0.24), `Pressure9am` (−0.23).

### 2. Models
| Model | Details |
|-------|---------|
| **LightGBM** | `LGBMClassifier` with default parameters |
| **XGBoost** | `XGBClassifier`, tracking log loss and error per boosting round |
| **Random Forest** | `RandomForestClassifier` with default parameters |
| **KNN** | `KNeighborsClassifier(n_neighbors=7)` |
| **Gaussian Naive Bayes** | `GaussianNB` |
| **Voting ensemble** | Hard voting over XGBoost, KNN, LightGBM, Random Forest and Naive Bayes |
| **RNN** | 1 GRU layer (32 units, recurrent dropout 0.2) → Dense(1); 30-day input windows, 15 epochs, Adam (AMSGrad) |
| **CNN** | 4 Conv1D layers (16 → 32 → 64 → 128 filters) with batch norm and max pooling, L1/L2 regularization; 30-day windows, 15 epochs |

Trained models are saved with `joblib` (`xgbmodel.sav`, `lgb_model.sav`, `k.sav`, `vote.sav`).

---

## Repository Structure

```
.
├── Term_Rainfall.ipynb   # Full pipeline: preprocessing, models, ensemble, RNN, CNN
├── DatasetAUS.csv        # Rain in Australia dataset (download from Kaggle)
├── xgbmodel.sav          # Saved XGBoost model
├── lgb_model.sav         # Saved LightGBM model
├── k.sav                 # Saved KNN model
├── vote.sav              # Saved voting ensemble
└── README.md
```

---

## Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn, LightGBM, XGBoost
- **Deep learning:** TensorFlow / Keras
- **Model saving:** joblib, pickle

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/brijeshpal/<repo-name>.git
   cd <repo-name>
   ```

2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost tensorflow optuna joblib jupyter
   ```

   The notebook imports `optuna.integration.lightgbm`. With recent Optuna versions this also needs `pip install optuna-integration`, or you can remove those import lines since they are not used.

3. Download the **Rain in Australia** dataset from Kaggle and save it as `DatasetAUS.csv` in the project folder.

4. Open and run the notebook:

   ```bash
   jupyter notebook Term_Rainfall.ipynb
   ```

   Some cells save files to hard-coded paths such as `D:/Project/` and `drive/MyDrive/`. Update these to your own folders.

---

## Limitations

- The train/test split uses `shuffle=False`, and the data is ordered by location, so the test set mostly contains stations not seen during training.
- Missing values are filled and features are standardized using statistics from the full dataset, before the split.
- The class imbalance (~80% "no rain") is not handled, which limits recall on rainy days.
- The RNN and CNN are trained with MSE loss for a binary target; binary cross-entropy with a sigmoid output would be a better fit.

---

## Conclusion

Advanced machine learning techniques, in particular **LightGBM** and ensemble classifiers, show strong potential for rainfall prediction. LightGBM achieved the highest accuracy while training quickly on a large dataset, which makes it a practical choice for real-world weather prediction systems.

---

## Future Work

- Use more diverse data sources, such as an **Indian weather dataset**.
- Integrate the models with climate change projections.
- Validate and adapt the models across different regions.
- Deploy the model for **real-time prediction**.
- Extend the work to **flood prediction**.
- Handle class imbalance (class weights, SMOTE) and tune hyperparameters with Optuna.

---

## Contact

**Brijesh Kumar**
- Email: [brijeshpal8878@gmail.com](mailto:brijeshpal8878@gmail.com)
- GitHub: [brijeshpal](https://github.com/brijeshpal)
- LinkedIn: [brijeshkumar8878](https://www.linkedin.com/in/brijeshkumar8878)
- Homepage: [brijeshpal.github.io](https://brijeshpal.github.io)
