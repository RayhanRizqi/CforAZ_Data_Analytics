import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Assuming the data is loaded as per previous instructions
data = pd.read_csv("CforAZ_Data_Analytics\processed_CFAZ Modeling Data.csv")
data_cleaned = data.drop(columns=['Unnamed: 0'])
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_cleaned), columns=data_cleaned.columns)
y = data_imputed['Total Contributions'].values

# Function to categorize contributions into classes
def categorize_contributions(contributions):
    bins = [0, 500, 1000, 2000, 7000, np.inf]  # Include np.inf to handle contributions above 3300
    labels = [0, 1, 2, 3, 4]  # The last category is for > 3300
    return np.digitize(contributions, bins, right=False) - 1

y_categorical = categorize_contributions(y)

# Visualize the distribution of categories
plt.figure(figsize=(10, 6))
plt.hist(y_categorical, bins=np.arange(6) - 0.5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(5), ['0-500', '500-1000', '1000-2000', '2000-7000', '>7000'])
plt.xlabel('Contribution Categories ($)')
plt.ylabel('Number of Contributions')
plt.title('Distribution of Contribution Categories')
plt.grid(True)
plt.show()
