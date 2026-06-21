import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from learning_model import LoanApprovalModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import random
import joblib


def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df = df.drop('loan_id', axis=1)
    df["education"] = df["education"].str.strip()
    df["self_employed"] = df["self_employed"].str.strip()
    df["loan_status"] = df["loan_status"].str.strip()

    df["education"] = df["education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    df["self_employed"] = df["self_employed"].map({
        "Yes": 1,
        "No": 0
    })

    df["loan_status"] = df["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)


    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
             train_dataset,
             batch_size=32,
            shuffle=True
            )
    
    return train_loader, X_val, y_val, X_test, y_test, scaler

def train_model(train_loader, X_val, y_val, X_test, y_test):
    model = LoanApprovalModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 49

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs.squeeze(), y_test)
    
    print(f"Test Loss: {test_loss.item():.4f}")

    return model

def save_model(model, scaler, model_path='loan_approval_model.pth', scaler_path='scaler.pkl'):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path} and scaler saved to {scaler_path}")

def predict_loan_approval(model, scaler, X_test, y_test):
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float()
        accuracy = (
            predictions.squeeze() == y_test
        ).float().mean()
    print(
        f"Test Accuracy: {accuracy.item()*100:.2f}%"
    )   
        
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    df = pd.read_csv('loan_approval_dataset.csv')
    train_loader, X_val, y_val, X_test, y_test, scaler = preprocess_data(df)
    model = train_model(train_loader, X_val, y_val, X_test, y_test)
    save_model(model, scaler)
    predict_loan_approval(model, scaler, X_test, y_test)
    
if __name__ == "__main__":
    main()