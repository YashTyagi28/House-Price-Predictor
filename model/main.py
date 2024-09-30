import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

def create_model(data):
    scaler=StandardScaler()
    X=data.drop(["SalePrice"],axis=1)
    y=np.log1p(data["SalePrice"])
    X = scaler.fit_transform(X)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y.values).float().unsqueeze(1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    
    class PricePredict(nn.Module):
        def __init__(self, input_features, output_features, hidden_units):
            super().__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_units),
                nn.Linear(in_features=hidden_units, out_features=hidden_units),
                nn.Linear(in_features=hidden_units, out_features=output_features),
            )
        def forward(self, x):
            return self.linear_layer_stack(x)
        
    model_1 = PricePredict(input_features=73,output_features=1,hidden_units=32)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)
    torch.manual_seed(42)
    epochs = 200

    for epoch in range(1,epochs+1):
        model_1.train()
        y_logits = model_1(X_train).squeeze()
        loss = loss_fn(y_logits, y_train.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_1.eval()
        with torch.inference_mode():
            test_logits = model_1(X_test).squeeze()
            test_loss = loss_fn(test_logits, y_test.squeeze())
        # if epoch % 50 == 1 or epoch == epochs:
        #     print(f"Epoch: {epoch} | Train Loss: {loss:.5f} | Test Loss: {test_loss:.5f}")
    return model_1,scaler

def encode_data(data, label_encoders):
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna("missing")
        if col in label_encoders:
            data[col] = label_encoders[col].transform(data[col])
        else:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    return data

def get_clean_data(path):
    data=pd.read_csv(path)
    data=data.drop(["Id","Alley","MasVnrType","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)
    for col in data.select_dtypes(include=['object']).columns:
        if not data[col].isnull().all():
            random_index = random.choice(data.index[data[col].notnull()])
            data.at[random_index, col] = "missing"
    label_encoders = {}
    data = encode_data(data, label_encoders)
    missing_cols = data.select_dtypes(include=['float', 'int']).isnull().sum()[data.isnull().sum() > 0]
    for col in missing_cols.index:
        data[col] = data[col].fillna(data[col].median())
    data=data.astype(float)
    return data,label_encoders

def test_preprocess(path,scaler,encoders):
    data=pd.read_csv(path)
    data=data.drop(["Alley","MasVnrType","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1) 
    data = encode_data(data,encoders)
    missing_cols = data.select_dtypes(include=['float', 'int']).isnull().sum()[data.isnull().sum() > 0]
    for col in missing_cols.index:
        data[col] = data[col].fillna(data[col].median())
    data=data.astype(float)
    X=data.drop(["Id"],axis=1)
    X = scaler.transform(X)
    X = torch.from_numpy(np.array(X)).float()
    return X,data["Id"].astype(int)

def main():
    data,encoders=get_clean_data("data/train.csv")
    model,scaler=create_model(data)
    X_test,id=test_preprocess("data/test.csv",scaler,encoders)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_pred = np.expm1(test_logits.numpy())
    submission = pd.DataFrame({
        'Id': id,
        'SalePrice': test_pred
    })
    submission.to_csv("data/submission.csv", index=False)
    print(submission["Id"])

    
if __name__=="__main__":
    main()
