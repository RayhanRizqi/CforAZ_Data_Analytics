import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

# Load and preprocess data
data = pd.read_csv("processed_CFAZ Modeling Data.csv")
data_cleaned = data.drop(columns=['Unnamed: 0'])
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_cleaned), columns=data_cleaned.columns)
X = data_imputed.drop(columns=['Total Contributions']).values
y = data_imputed['Total Contributions'].values

# Define the bins and labels for classification
bins = [0, 500, 1000, 2000, 7000]
labels = [0, 1, 2, 3]  # Classes corresponding to the intervals
y_binned = np.digitize(y, bins) - 1

# Normalize feature data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Convert arrays to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_binned, dtype=torch.long)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
print(X_train.shape[1])
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the neural network model for classification
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.layer2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.layer3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 4)  # Output layer for 5 classes

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

model = ClassificationModel()
criterion = nn.CrossEntropyLoss()
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

# Evaluate and display results
model.eval()
class_ranges = ['0-500', '500-1000', '1000-2000', '2000-7000']
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        for p, t in zip(predicted.numpy(), target.numpy()):
            actual_class = class_ranges[t]
            predicted_class = class_ranges[p]
            correct = 'Correct' if t == p else 'Incorrect'
            print(f"Predicted: {predicted_class}, Actual: {actual_class}, Result: {correct}")

# Gets the state of the model
model_state = model.state_dict()

# Saves the model state to model_weights.model
torch.save(model_state, 'CforAZ_Data_Analytics/model_weights.model')

# # Evaluate and print confusion matrix
# model.eval()
# all_preds = []
# all_targets = []
# with torch.no_grad():
#     for data, target in val_loader:
#         output = model(data)
#         _, predicted = torch.max(output, 1)
#         all_preds.extend(predicted.numpy())
#         all_targets.extend(target.numpy())

# # Assuming 'all_preds' and 'all_targets' are the lists of predictions and true labels from your validation set
# f1_scores_per_class = f1_score(all_targets, all_preds, labels=[0, 1, 2, 3], average=None)
# f1_score_average = f1_score(all_targets, all_preds, average='weighted')

# print(f"F1 score per class: {f1_scores_per_class}")
# print(f"Average F1 score: {f1_score_average}")

# # Calculate confusion matrix
# cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
# print("Confusion Matrix:")
# print(cm)

# # Plot confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()