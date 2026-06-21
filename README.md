# PyTorch Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved using a Feed Forward Neural Network built with PyTorch.

## Project Overview

This project uses financial and applicant information to classify loan applications as either:

* Approved
* Rejected

The complete machine learning pipeline includes:

* Data preprocessing
* Feature engineering
* Data standardization
* Neural network training
* Model evaluation
* Model persistence
* Prediction on new applicants

## Dataset Features

The model uses the following features:

1. Number of Dependents
2. Education Status
3. Self Employment Status
4. Annual Income
5. Loan Amount
6. Loan Term
7. CIBIL Score
8. Residential Asset Value
9. Commercial Asset Value
10. Luxury Asset Value
11. Bank Asset Value

### Target Variable

* Loan Approved → 1
* Loan Rejected → 0

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-Learn
* Joblib

## Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns
* Handled categorical variables using label mapping
* Train/Validation/Test split
* Feature standardization using StandardScaler
* Conversion to PyTorch tensors

## Neural Network Architecture

Input Layer: 11 Features

Hidden Layer 1:

* Linear Layer
* ReLU Activation

Hidden Layer 2:

* Linear Layer
* ReLU Activation

Output Layer:

* Single Neuron

Loss Function:

* BCEWithLogitsLoss

Optimizer:

* Adam

Learning Rate:

* 0.001

Epochs:

* 50

Batch Size:

* 32

## Model Performance

Test Accuracy: ~97%

## Project Structure

loan-approval-neural-network/

├── learning_model.py

├── training_model.py

├── testing.py

├── loan_approval_dataset.csv

├── loan_approval_model.pth

├── scaler.pkl

├── README.md

└── requirements.txt

## Running the Project

Install dependencies:

pip install -r requirements.txt

Train the model:

python training_model.py

Run predictions:

python testing.py

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Early stopping
* Model explainability
* Deployment using Flask or FastAPI

## Author

Prajeet

Engineering Physics, IIT Ropar
