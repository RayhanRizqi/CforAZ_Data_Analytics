import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

data = pd.read_csv("CforAZ_Data_Analytics\processed_CFAZ Modeling Data.csv")
# Drop the 'Unnamed: 0' column
data_cleaned = data.drop(columns=['Unnamed: 0'])

# Check for missing values and fill with the mean of each column
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_cleaned), columns=data_cleaned.columns)

# Extract features and target
X = data_imputed.drop(columns=['Total Contributions'])
y = data_imputed['Total Contributions']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train.shape, X_val.shape

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(64, activation='relu'),  # Second hidden layer
    Dense(1)  # Output layer for regression
])

# Compile the model with the Adam optimizer and mean squared error loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
final_loss = model.evaluate(X_val, y_val)
print("Final validation loss:", final_loss)
