# 🩺 Diabetes Probability Prediction – Midterm Project  
Machine Learning Zoomcamp – Midterm Submission  
**Author: Alisa Su**

---

## 1. Problem Description & Context

Diabetes is a growing global health concern affecting millions worldwide. Early detection and risk assessment can prevent complications such as cardiovascular disease, kidney failure, neuropathy, and vision loss.

This project predicts the **probability that an individual has diabetes** based on demographic, lifestyle, and medical health indicators.

### 🎯 Goal
Build a machine learning model and deployment pipeline that:

- Accepts a person's health attributes  
- Predicts diabetes probability (`0–1`)  
- Returns a binary prediction (`diabetes = True/False`)  
- Can be accessed through a **Flask web service**  
- Can be deployed with **Docker**

### 💡 Use Case
Healthcare apps or wellness platforms can integrate the model to:

- Flag high-risk individuals  
- Provide early lifestyle advice  
- Prioritize medical follow-up  
- Support preventive care

This project is for educational and research purposes only.

---

## 2. Dataset Overview

- Dataset contains **no missing values**  
- All variables represent lifestyle or health indicators
- Target column: **`diabetes_binary`**  
  - `0` → No diabetes  
  - `1` → Diabetes

---

## 3. Exploratory Data Analysis (EDA)

### ✔ Numerical Features – Correlation Coefficient
Correlation analysis showed moderate relationships between:

- BMI → diabetes  
- Age → diabetes  
- Physical health score → diabetes  

### ✔ Categorical Features – Mutual Information
Mutual Information (MI) identified the most informative predictors:

- General health  
- High blood pressure  
- High cholesterol  
- Smoking  
- Physical activity  

### 🔑 Key Insights
- Higher BMI strongly increases diabetes likelihood  
- Cardiovascular risk factors align heavily with diabetes risk  
- Lifestyle choices (smoking, inactivity) are strong indicators  
- Dataset was clean and ready for modeling (no NA handling needed)

---

## 4. Model Training & Experimentation

Three machine learning models were trained and evaluated:

### **4.1 Logistic Regression**
- 5-Fold Cross-Validation
- Tuned parameter: **C**
- Evaluated using ROC AUC
- Served as baseline model

---

### **4.2 Decision Tree Classifier**
Hyperparameters tuned:

- `max_depth`  
- `min_samples_leaf`

Findings:
- Very deep trees overfitted  
- Balanced depth improved generalization  

---

### **4.3 Random Forest Classifier**
Hyperparameters tuned:

- `n_estimators`  
- `max_depth`  
- `min_samples_leaf`

Random Forest delivered the **highest ROC AUC** and became the final model.

---

## 5. Model Saving & Loading

Model artifacts (`DictVectorizer` + model) were saved using pickle:

```python
with open('rf_model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)
```

Loaded in the API:

```python
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
```

---

## 6. Flask API Deployment

API endpoint:
```
POST /predict
```

Input:  
JSON object containing a patient’s health features

Output example:
```json
{
  "diabetes": false,
  "diabetes_probability": 0.137
}
```

---

## 7. Environment Setup & Dependencies

### **Using Pipenv**
```bash
pipenv install
pipenv shell
```

### **Using requirements.txt**
```bash
pip install -r requirements.txt
```

---

## 8. Docker Deployment

### **Build Docker Image**
```bash
docker build -t midterm-project .
```

### **Run Container**
```bash
docker run -p 9696:9696 midterm-project
```

API will be available at:
```
http://localhost:9696/predict
```

---

## 9. Testing the API

`predict_test.py` sends a sample request:

```python
import requests

url = "http://localhost:9696/predict"

patient = {
    "BMI": 28,
    "Age": 9,
    "Smoking": 1,
    ...
}

response = requests.post(url, json=patient).json()
print(response)
```

---

## 10. Project Structure

```
07-midterm-project/
│
├── train.py
├── predict.py
├── predict_test.py
├── rf_model.bin
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 11. Video Demo
*(Insert Loom or YouTube link here)*

---

## 12. Conclusion

This midterm project demonstrates a complete end-to-end ML pipeline:

- Data exploration  
- Model training and hyperparameter tuning  
- Saving/loading models  
- Deploying a prediction API with Flask  
- Dockerizing the entire application  

The **Random Forest model** provided the strongest performance and is deployed as the final API.

