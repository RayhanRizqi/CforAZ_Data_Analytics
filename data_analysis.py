import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('processed_CFAZ Modeling Data.csv')

ax = df.plot(x='Max Give', y='Log_Total Contributions', marker='o', title='Line Chart')
ax.set_xlabel("X Values")
ax.set_ylabel("Y Values")
plt.grid(True)
plt.show()