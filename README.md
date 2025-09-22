# 🚗 Predicting Price and Fuel Type of Used Cars 

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-yellow.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/HRNBEnninful/Project?style=social)](https://github.com/HRNBEnninful/Project/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HRNBEnninful/Project?style=social)](https://github.com/HRNBEnninful/Project/network)

---

## 🔍 Description
This repository contains machine learning pipelines for two tasks:
- **Regression Task** – Predicting the price of used cars based on vehicle attributes.  
- **Classification Task** – Predicting the fuel type of a car using multiple classifiers.

---

## 🚀 Features
- Dataset from **Kaggle**: [Cars Dataset](https://www.kaggle.com/datasets/aishwaryamuthukumar/cars-dataset-audi-bmw-ford-hyundai-skoda-vw)  
- Exploratory Data Analysis  
- End-to-end ML pipelines for both regression and classification  
- Model comparison with multiple metrics  
- Feature importance analysis  

---

## 📊 Regression: Predicting Used Car Prices

**Input Features**  
- **Numerical:** year, mileage, tax, mpg, engineSize  
- **Categorical:** Make, model, transmission, fuelType  

**Target Variable**  
- `price` (log-transformed during training for stability)

**Models Implemented**  
- Ridge Regression (baseline linear model)  
- XGBoost Regressor (tree-based boosting model)  
- Artificial Neural Network (ANN) (Keras-based feedforward NN)

**Pipeline**  
```mermaid
flowchart LR
    Data[cars_dataset.csv] --> Preprocessing
    Preprocessing --> Ridge
    Preprocessing --> XGBoost
    Preprocessing --> ANN

Hyperparameter tuning:
- Ridge → Grid Search
- XGBoost → Randomized Search
- ANN → Randomized hyperparameter sampling with early stopping

Evaluation metrics:
MAE, RMSE, MAPE, R²

Feature importance extracted from each model (coefficients, gain importance, permutation importance).

Outputs:
- Trained models saved (.pkl and .keras)
- Performance summary (performance_summary.csv)
- Top 10 important features for each model (ridge_top10_features.csv, xgboost_top10_features.csv, ann_top10_features.csv)

| Model   | MAE (\$) | RMSE (\$) | MAPE (%) | RMSE (%) | R²    | Best Hyperparameters                                                                         |
| ------- | -------- | --------- | -------- | -------- | ----- | -------------------------------------------------------------------------------------------- |
| Ridge   | 1,559    | 2,556     | 9.40     | 12.69    | 0.926 | `alpha=0.1`                                                                                  |
| XGBoost | 1,413    | 2,410     | 8.44     | 11.45    | 0.934 | `subsample=1.0, n_estimators=400, max_depth=5, learning_rate=0.05, colsample_bytree=1.0`     |
| ANN     | 1,291    | 2,277     | 7.82     | 10.57    | 0.941 | `(neurons1=64, neurons2=32, dropout=0.1, lr=0.001, batch=64, activation='tanh', epochs=100)` |

✅ Notes:
- MAE = Mean Absolute Error
- RMSE = Root Mean Squared Error
- MAPE = Mean Absolute Percentage Error
- R² = Coefficient of Determination

---

## 🔎 Classification: Predicting Fuel Type

Input Features:
Numerical: year, mileage, tax, mpg, engineSize
Categorical: Make, model, transmission, fuelType

Target Variable:
fuelType (categorical classification)

Models Implemented:
- Logistic Regression
- Random Forest
- Neural Network (MLPClassifier)
- CatBoost Classifier
- Stacking Ensemble (combines the above models)

**Pipeline**
```mermaid
flowchart LR
    Data[cars_dataset.csv] --> Preprocessing
    Preprocessing --> RF[Random Forest]
    Preprocessing --> LR[Logistic Regression]
    Preprocessing --> NN[Neural Network]
    Preprocessing --> CatBoost
    RF & LR & NN & CatBoost --> Stacking[Stacking Ensemble]

Key Steps:

- Handle class imbalance using SMOTENC.
- Preprocessing pipeline (scaling + one-hot encoding).
- Hyperparameter tuning with RandomizedSearchCV.

Evaluation metrics:
Accuracy, Precision, Recall, F1-score

Feature importance: 
Permutation importance analysis for each model.

Outputs:
- Trained models saved (rf_fueltype_model.pkl, lr_fueltype_model.pkl, nn_fueltype_model.pkl, cat_fueltype_model.pkl, stacking_fueltype_model.pkl, fueltype_preprocessor.pkl)
- Performance summary: fueltype_model_performance.csv
- Confusion matrices saved: e.g., random_forest_confusion_matrix.csv

Feature importances:
- Individual CSVs per model (e.g., random_forest_permutation_feature_importances.csv)
- Combined summary: all_models_permutation_feature_importances.csv

| Model               | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Random Forest       | 0.9882   | 0.65     |
| Logistic Regression | 0.8796   | 0.54     |
| Neural Network      | 0.9731   | 0.61     |
| CatBoost            | 0.9414   | 0.59     |
| Stacking Ensemble   | 0.9886   | 0.65     |

⚠️ Note: 'Electric' and 'Other' classes are very underrepresented, which impacts precision/recall for these rare classes.

---

## 🚀 How to Run

Place your dataset as cars_dataset.csv in the project root.
Open the jupyter notebook: Used_Cars.ipynb
Outputs (CSV files and saved models) will appear in the project folder.

---

## 📦 Dependencies

- Python 3.9+
- pandas, numpy, 
- scikit-learn, imbalanced-learn
- XGBoost / Catboost
- Tensorflow / Keras
- plotly, seaborn, matplotlib

Install all requirements:
pip install -r requirements.txt

---

## 📊 Results

- Ridge regression serves as a strong linear baseline.
- XGBoost generally outperforms Ridge with better handling of nonlinear relationships.
- ANN can match or exceed performance with tuned hyperparameters.

For classification, ensemble methods (Random Forest, CatBoost, Stacking) typically achieve the best performance across accuracy and F1-score.

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/HRNBEnninful/Used_Cars_Project.git
cd Used_Cars
pip install -r requirements.txt
jupyter notebook Used_Cars.ipynb
```

---

## 🤝 Contributing
Contributions are welcome! 
Please:
- Fork the repository
- Create a feature branch (git checkout -b feature-newmodel)
- Commit changes and push
- Open a Pull Request

## 📜 License
This project is licensed under the MIT License – feel free to use and modify.

## 📬 Contact
Henry Reynolds Nana Benyin Enninful
📧 Email: hrnbenninful@gmail.com
🐙 GitHub: @HRNBEnninful
💼 LinkedIn: https://www.linkedin.com/in/henryrnbenninful/
