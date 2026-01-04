## 🍎 Fruit Classifier – Deep Learning Image Classification Project
## 📌 Problem Description

The goal of this project is to build an image classification system that can automatically identify different types of fruits from images.

This solution can be used in scenarios such as:

Automated fruit recognition in retail or smart agriculture

Educational demonstrations of computer vision and transfer learning

A lightweight example of deploying deep learning models in containers

The project focuses on simplicity, correctness, and reproducibility, while demonstrating best practices in deep learning model training, validation, and deployment.

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

The EDA focuses on understanding the visual and structural properties of the image data:

Image dimensions and consistency

RGB color distribution across classes

Visual inspection of sample images per class

Confirmation that labels match image content

Additional checks were performed to ensure:

No duplicated images

No overlap between training and validation sets

This confirms that the high model accuracy is due to dataset simplicity, not data leakage or flawed methodology.

## 🧠 Model Training

All experiments and training steps are implemented in:

fruits_classifier_xception.ipynb

Key Training Steps
1. Pre-trained Convolutional Neural Networks

Used pre-trained CNNs to leverage learned visual features

Reduced training time and improved stability

2. Transfer Learning

Base model weights loaded from ImageNet

Final layers fine-tuned on the fruit dataset

3. Learning Rate Tuning

Multiple learning rates tested

Best learning rate found: 0.01

4. Inner Dense Layer Size

Tested different dense layer sizes

Best size: 100

5. Dropout Regularization

Dropout values evaluated

Best dropout: 0.0
(Dataset is simple and does not require regularization)

6. Data Augmentation

Applied to improve generalization

Includes random flips, rotations, and shifts

⚠️ Model Evaluation Notes

Because the Fruits-360 dataset is visually simple and well-separated, the model reaches:

val_accuracy = 1.0 after epoch 1

⚠️ Important:
Instead of accuracy, val_loss is used as the main monitoring metric to:

Prevent misleading conclusions

Better capture model confidence and stability

## ✅ Model Validation & Testing

Explicit checks confirm:

No duplicated samples

No train/validation overlap

High accuracy is purely due to dataset quality

Final best model is saved and tested manually

Manual Test Result

A downloaded image from test folder was used, and the model correctly predicted:

"Mango"

## 📦 Environment Setup
1. Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate        

2. Install Dependencies
pip install -r requirements.txt

Main dependencies include:

TensorFlow / Keras

NumPy

Pandas

Pillow

Flask (for serving)

Gunicorn

## 🐳 Containerization (Docker)

The application is fully containerized for easy deployment.

Build Docker Image
docker build -t fruit-classifier .

Run Container
docker run -p 8080:8080 fruit-classifier

Once running, the service will be available at:

http://localhost:8080

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

