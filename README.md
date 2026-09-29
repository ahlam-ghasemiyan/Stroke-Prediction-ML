# 🧠 Stroke Prediction ML

![Banner](backg (1).png)

> A machine learning project for predicting stroke risk using Logistic Regression and Random Forest, with a complete preprocessing pipeline, SMOTE, and saved models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)](https://imbalanced-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 About the Project

This project is a machine learning system for **stroke prediction**.  
It works with the `Stroke_dataset.csv` dataset and builds models using a complete pipeline that includes:

- Missing value handling
- Standardization of numerical features
- One-Hot Encoding of categorical features
- Class balancing with **SMOTE**
- Training **Logistic Regression** and **Random Forest** models
- Saving the final models in `stroke_models.pkl`

---

## ✨ Features

- 🧹 Complete data preprocessing
- ⚖️ SMOTE for handling class imbalance
- 🤖 Two machine learning models: Logistic Regression and Random Forest
- 💾 Saved full pipelines in a single `.pkl` file
- 📊 Reusable for prediction on new data
- 🖼️ Ready-to-use banner and GitHub-friendly structure

---

## 📊 Dataset

Dataset file: `Stroke_dataset.csv`

Main features:

| Feature | Description |
|---|---|
| gender | Gender |
| age | Age |
| hypertension | Hypertension status |
| heart_disease | Heart disease status |
| ever_married | Marital status |
| work_type | Work type |
| Residence_type | Residence type |
| avg_glucose_level | Average glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking status |
| stroke | Target variable: stroke occurrence |

---

## 🧪 Models & Pipeline

The saved models in `stroke_models.pkl` include the following steps:

1. `SimpleImputer` for missing values
2. `StandardScaler` for numerical feature scaling
3. `OneHotEncoder` for categorical feature encoding
4. `SMOTE` for class balancing
5. Final models:
   - Logistic Regression
   - Random Forest

> The `.pkl` file is stored as a dictionary-like object.  
> You can access the models with the keys `Logistic Regression` and `Random Forest`.

---

## 📁 Project Structure

```text
Stroke-Prediction-ML/
├── Stroke_dataset.csv
├── stroke_models.pkl
├── banner.png
├── README.md
└── requirements.txt
If you keep the original image name, use this in the README instead of banner.png:
![Banner](backg%20(1).png)

⚙️ Installation
Clone the repository:

bash
git clone https://github.com/ahlam-ghasemiyan/Stroke-Prediction-ML.git
cd Stroke-Prediction-ML
Install the required packages:

bash
pip install -r requirements.txt
Suggested requirements.txt:

text
pandas
numpy
scikit-learn==1.6.1
imbalanced-learn
joblib
matplotlib
seaborn
The saved model was created with scikit-learn 1.6.1.
Using the same version is recommended to avoid compatibility warnings.

🚀 Usage Example
python
import joblib
import pandas as pd

# Load models
models = joblib.load("stroke_models.pkl")

log_reg = models["Logistic Regression"]
rand_forest = models["Random Forest"]

# New sample data
new_data = pd.DataFrame([{
    "gender": 1,
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": 1,
    "work_type": "Private",
    "Residence_type": 1,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}])

# Prediction
prediction = log_reg.predict(new_data)[0]
probability = log_reg.predict_proba(new_data)[0, 1]

print("Prediction:", prediction)
print("Probability:", probability)

🖼️ Screenshot
https://backg (1).png/

🛠️ Technologies
Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn

Joblib

Matplotlib / Seaborn


📄 License
This project is licensed under the MIT License.

📬 Contact
GitHub: ahlam-ghasemiyan
