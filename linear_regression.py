import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('processed_CFAZ Modeling Data.csv')

# Drop the index column
df = df.drop(columns=['Unnamed: 0'])

# Splitting the data into input and output
X = df.drop(columns=['Total Contributions'])  # Inputs
y = df['Total Contributions']  # Output

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the linear regression model
model = LinearRegression()

for epoch in range(100):
    print("Epoch:", epoch + 1)
    # Training the model
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)
    if epoch == 0 or epoch == 99:
        print(y_pred)
        for i in range(len(y_test)):
            y_pred[i] = max(0,y_pred[i])
            print("predicted: " + str(y_pred[i]) + " and expected: " + str(y_test.iloc[i]))

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

print("MSE: " + str(mse))
print("r2: " + str(r2))
