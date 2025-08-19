# ML-Model-Inference-with-FastAPI
✅Iris Classification API

✅Problem Description

This project implements a FastAPI application for classifying Iris flower species (setosa, versicolor, virginica) based on four numerical features: sepal length, sepal width, petal length, and petal width. The model is trained on the Iris dataset provided in Iris.csv.

✅Model Choice Justification

A RandomForestClassifier was chosen due to its:

Robust performance on small datasets

Ability to handle non-linear relationships

Good generalization with minimal tuning

Provision of probability scores for predictions

The model is trained with standardized features using StandardScaler to ensure consistent input preprocessing.

✅API Usage Examples

1. Health Check

GET http://localhost:8000/
Response:
{
    "status": "healthy",
    "message": "Iris Classification API is running"
}

2. Prediction

POST http://localhost:8000/predict
Body:
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
Response:
{
    "prediction": "Iris-setosa",
    "confidence": 0.98
}

✅How to Run the Application

✅Clone the repository

✅Install dependencies:

pip install -r requirements.txt

✅Ensure model.pkl, scaler.pkl, and Iris.csv are in the project directory

Run the model training notebook (iris_classification_detailed.ipynb) to generate model.pkl and scaler.pkl if not already present

✅Run the FastAPI application:

uvicorn main:app --reload





Access the API documentation at http://localhost:8000/docs

Assumptions and Limitations


The model assumes input features are within the range of the training data

Input validation ensures float values but doesn't check for realistic biological ranges



The model is trained on a small dataset (150 samples), which may limit generalization

The API uses standardized features, requiring the scaler to be loaded

✅Project Structure

iris_classification_detailed.ipynb: Notebook for data exploration, model training, and evaluation

main.py: FastAPI application

model.pkl: Saved RandomForestClassifier model

scaler.pkl: Saved StandardScaler for preprocessing

requirements.txt: Project dependencies

README.md: Project documentation
