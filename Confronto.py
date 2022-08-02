import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    f1_score,
)

from preprocessing import TEST_SIZE
from MLFineTuning import RANDOM_STATE

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Funzione per la stampa a console e la creazione di una confusion matrix dei risultati di un modello
def performance(model_name, y_test, y_pred, type, graph=True):
    print(
        "-----------------",
        model_name,
        "-----------------",
    )

    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if graph:
        plt.savefig("./Images/Results/" + model_name + "_" + type + ".png")

    print("----------------------------------------------------------")
    print("Confusion Matrix")
    print(cm)
    print("----------------------------------------------------------")
    print("Classification Report")
    print(classification_report(y_pred, y_test))


# Script che crea, allena e testa gli stessi due modelli prima sul dataset puro, poi sulla sua versione processata
def main():
    df = pd.read_csv("./Datasets/Adult.csv")
    y = df.pop("income")
    X = df

    o = OrdinalEncoder()
    X = o.fit_transform(X)

    l = LabelEncoder()
    y = l.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    XGmodel = XGBClassifier(criterion="gini", max_depth=5, max_features="log2", n_estimators=140)
    RFmodel = RandomForestClassifier(
        bootstrap=True,
        max_depth=20,
        max_features=6,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=120,
    )

    XGmodel.fit(X_train, y_train)
    RFmodel.fit(X_train, y_train)

    y_pred_XG = XGmodel.predict(X_test)
    y_pred_RF = RFmodel.predict(X_test)

    print("\n\nDATASET NON PREPROCESSATO\n\n")

    performance("XGBoost", y_test, y_pred_XG, "noP")
    performance("RandomForest", y_test, y_pred_RF, "noP")

    X_train = pd.read_csv("./Datasets/TrainData.csv")
    y_train = pd.read_csv("./Datasets/TrainLabel.csv")
    X_test = pd.read_csv("./Datasets/TestData.csv")
    y_test = pd.read_csv("./Datasets/TestLabel.csv")

    XGmodel.fit(X_train, y_train)
    RFmodel.fit(X_train, y_train)

    y_pred_XG = XGmodel.predict(X_test)
    y_pred_RF = RFmodel.predict(X_test)

    print("\n\nDATASET PREPROCESSATO\n\n")

    performance("XGBoost", y_test, y_pred_XG, "noP", False)
    performance("RandomForest", y_test, y_pred_RF, "noP", False)


if __name__ == "__main__":
    main()
