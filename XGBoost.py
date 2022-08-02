import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from xgboost import XGBClassifier
from Confronto import performance

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Script che si occupa di allenare sul training e validare sul test XGBoost
def main():
    X_train = pd.read_csv("./Datasets/TrainData.csv")
    y_train = pd.read_csv("./Datasets/TrainLabel.csv")
    X_test = pd.read_csv("./Datasets/TestData.csv")
    y_test = pd.read_csv("./Datasets/TestLabel.csv")

    MLmodel = XGBClassifier(criterion="gini", max_depth=6, max_features="log2", n_estimators=100)
    MLmodel.fit(X_train, y_train)
    y_pred_ML = MLmodel.predict(X_test)

    performance("XGBoost", y_test, y_pred_ML, "P")


if __name__ == "__main__":
    main()
