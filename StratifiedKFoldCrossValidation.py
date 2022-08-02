from numpy import mean
import pandas as pd
from numpy import std

# Removing useless Warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def main():

    X_train = pd.read_csv("./Datasets/TrainData.csv")
    y_train = pd.read_csv("./Datasets/TrainLabel.csv")

    y_train = y_train.values.ravel()

    # Oggetto necessario per la cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Array dei modelli da provare
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        XGBClassifier(use_label_encoder=False),
    ]
    # Valutazione del modello tramite f1-score cross validato
    for i in range(0, len(models)):
        scores = cross_val_score(
            models[i], X_train.values, y_train, scoring="f1_micro", cv=cv, n_jobs=-1
        )
        # Stampa delle prestazioni
        print(str(models[i]) + " Score: %.3f (%.3f)" % (mean(scores), std(scores)))


if __name__ == "__main__":
    main()
