import torch
import numpy as np
import joblib
from learning_model import LoanApprovalModel
import pandas as pd

scaler = joblib.load("scaler.pkl")
customer = pd.DataFrame([{
    "no_of_dependents": 2,
    "education": 1,
    "self_employed": 0,
    "income_annum": 5000000,
    "loan_amount": 1000000,
    "loan_term": 12,
    "cibil_score": 750,
    "residential_assets_value": 2000000,
    "commercial_assets_value": 500000,
    "luxury_assets_value": 300000,
    "bank_asset_value": 1000000
}])
customer = scaler.transform(customer)

model = LoanApprovalModel()
model.load_state_dict(torch.load("loan_approval_model.pth"))
model.eval()
customer = torch.tensor(customer, dtype=torch.float32)

with torch.no_grad():
    output = model(customer)

probability = torch.sigmoid(output).item()
print(f"Approval Probability: {probability:.4f}")

if probability >= 0.5:
    print("Loan Approved")
else:
    print("Loan Rejected")