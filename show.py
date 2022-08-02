import pandas as pd
import matplotlib.pyplot as plt


y_train = pd.read_csv("./Datasets/TestLabel.csv")

plt.title(label="Income frequency visualization - Pie Chart")
y_train["income"].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()