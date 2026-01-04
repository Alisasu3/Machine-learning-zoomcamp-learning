## 🍎 Fruit Classifier – Deep Learning Image Classification Project
---
## 📌 Problem Description

The objective of this project is to develop an image classification system capable of accurately identifying different types of fruits from images using deep learning techniques.

The project emphasizes simplicity, correctness, and reproducibility, while showcasing best practices in model training, validation, and deployment.

## 📊 Dataset
Source

This project is based on the Fruits-360 dataset:

Official repository:
👉 https://github.com/fruits-360/fruits-360

Subset Used

To reduce computational cost and speed up experimentation, only 5 common fruit classes were extracted to form a smaller dataset:

🍌 Banana

🍎 Apple Golden 1

🍊 Orange

🥭 Mango

🍓 Strawberry

This reduced dataset still provides very clear visual patterns, making it suitable for transfer learning experiments.

## 🔍 Exploratory Data Analysis (EDA)

The exploratory data analysis focuses on understanding both the visual characteristics and the structural integrity of the image dataset. Key aspects examined include:
- Image dimensions and overall consistency
- RGB color distribution across different fruit classes
- Visual inspection of representative samples from each class
- Verification that image labels correctly correspond to their visual content

In addition, data integrity checks were conducted to ensure:
- No duplicated images within the dataset
- No overlap between training and validation datasets

These checks confirm that the observed high model performance is attributable to the inherent simplicity and clarity of the dataset, rather than data leakage or methodological issues.

## 🧠 Model Training

All experiments and model training steps are implemented in the notebook:  
[`fruits_classifier_xception.ipynb`](fruits_classifier_xception.ipynb)

### Key Training Steps
### 1. Pre-trained Convolutional Neural Networks

Used pre-trained CNNs to leverage learned visual features

Reduced training time and improved stability

### 2. Transfer Learning

Base model weights loaded from ImageNet

Final layers fine-tuned on the fruit dataset

### 3. Learning Rate Tuning

Multiple learning rates tested

Best learning rate found: 0.01

### 4. Inner Dense Layer Size

Tested different dense layer sizes

Best size: 100

### 5. Dropout Regularization

Dropout values evaluated

Best dropout: 0.0
(Dataset is simple and does not require regularization)

### 6. Data Augmentation

Applied to improve generalization

Includes random flips, rotations, and shifts

### 7.  Model Saving and Testing

The best-performing model is saved using model checkpoints. A manual test is performed using a randomly selected image from the test dataset, and the model correctly predicts the image as "Mango".

## ⚠️ Model Evaluation Notes

Because the Fruits-360 dataset is visually simple and well-separated, the model reaches:

val_accuracy = 1.0 after epoch 1

**Important:**  
Instead of relying solely on accuracy, `val_loss` is used as the primary monitoring metric in order to:
- Avoid misleading performance interpretations
- Better capture model confidence and convergence stability

## 📦 Environment Setup

1. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```     
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Test endpoints
```bash
curl http://127.0.0.1:5000/health
```
4. Test prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -F "file=@../data/fruits/test/Mango/134_100.jpg"
```
<img width="775" height="217" alt="image" src="https://github.com/user-attachments/assets/8d247cd1-4fa3-4885-8966-85d70983c568" />

## 🐳 Containerization (Docker)

The application is fully containerized for easy deployment.

Build Docker Image
```bash
docker build -t fruit-classifier .
```
Run Container
```bash
docker run -p 5000:5000 fruit-classifier
```
Test
```bash
curl http://127.0.0.1:5000/health
```
<img width="766" height="165" alt="image" src="https://github.com/user-attachments/assets/4f939a30-e19f-4e5c-982b-79946b3b596b" />

Predict
```bash
	curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@../data/fruits/test/Orange/52_100.jpg"
```
<img width="763" height="217" alt="image" src="https://github.com/user-attachments/assets/081dd6eb-66a6-4b2f-a72e-5cf112f22a55" />

## 🚀 Project Highlights

✅ Clean and well-structured transfer learning workflow  
✅ Robust validation with explicit checks to prevent data leakage  
✅ Systematic hyperparameter tuning (learning rate, dense layer size, dropout)  
✅ Data augmentation applied to improve model robustness  
✅ Final model saved and manually verified using external images  
✅ Dependencies managed via a virtual environment  
✅ Model deployment implemented using a Flask API  
✅ Fully containerized with Docker for reproducible deployment 

## 📌 Conclusion

This project demonstrates how a well-structured deep learning pipeline, combined with transfer learning and proper validation, can achieve excellent performance—even on small datasets.

## 💬 Feedback

Feedback or suggestions are welcome. Please feel free to reach out.






