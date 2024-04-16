import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv("CforAZ_Data_Analytics\processed_CFAZ Modeling Data.csv")
data_cleaned = data.drop(columns=['Unnamed: 0'])
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_cleaned), columns=data_cleaned.columns)
X = data_imputed.drop(columns=['Total Contributions']).values
y = data_imputed['Total Contributions'].values.reshape(-1, 1)

# Normalize both features and target data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert arrays to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.layer2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

train_model(100)

# Evaluate and report final validation loss scaled back to original
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    final_loss = 0
    for data, target in val_loader:
        output = model(data)
        predictions.extend(output.view(-1).tolist())
        actuals.extend(target.view(-1).tolist())
        final_loss += criterion(output, target).item()
    final_loss /= len(val_loader)
    final_loss = scaler_y.inverse_transform([[final_loss]])[0][0]
print("Final validation loss:", final_loss)
# Scale predictions and actuals back to original
predictions_scaled = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals_scaled = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))

# Print predictions, actuals, and differences
for pred, actual in zip(predictions_scaled, actuals_scaled):
    difference = pred - actual
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}, Difference: {difference[0]:.2f}")
