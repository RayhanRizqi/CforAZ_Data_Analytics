import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

df = pd.read_csv('processed_CFAZ Modeling Data.csv')

# Drop the index column
df = df.drop(columns=['Unnamed: 0'])

# Splitting the data into input and output
X = df.drop(columns=['Log_Total Contributions'])  # Inputs
y = df['Log_Total Contributions']  # Output

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Lasso regression model with an alpha value
# Alpha is a hyperparameter and may need tuning
lasso_model = Lasso(alpha=0.01)

# Training the Lasso model
lasso_model.fit(X_train, y_train)

# Predicting on the test set
y_pred = lasso_model.predict(X_test)

# Evaluating the Lasso model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Lasso MSE: " + str(mse))
print("Lasso r2: " + str(r2))

# Displaying coefficients from Lasso model
for feature, coef in zip(X.columns, lasso_model.coef_):
    print(f'{feature}: {coef}')