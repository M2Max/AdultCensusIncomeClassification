import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier
from Confronto import performance

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Script che si occupa di allenare sul training e validare sul test RandomForest
def main():
    X_train = pd.read_csv("./Datasets/TrainData.csv")
    y_train = pd.read_csv("./Datasets/TrainLabel.csv")
    X_test = pd.read_csv("./Datasets/TestData.csv")
    y_test = pd.read_csv("./Datasets/TestLabel.csv")

    MLmodel = RandomForestClassifier(
        bootstrap=True,
        max_depth=19,
        max_features=7,
        min_samples_leaf=1,
        min_samples_split=15,
        n_estimators=110,
    )
    MLmodel.fit(X_train, y_train)
    y_pred_ML = MLmodel.predict(X_test)

    performance("RandomForestClassifier", y_test, y_pred_ML, "P")


if __name__ == "__main__":
    main()
