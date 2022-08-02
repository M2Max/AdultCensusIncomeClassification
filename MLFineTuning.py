import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
)

# Rimuove warning inutili
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Costante Random State
RANDOM_STATE = 42

# Lista di modelli da utilizzare
models = [
    RandomForestClassifier(),
    XGBClassifier(use_label_encoder=False),
]

# Stringhe con il nome dei modelli
model_name = ["RandomForestClassifier", "XGBClassifier"]

# Lista dei parametri per il Randomized search
random_parameters = [
    # Parametri RandomForest
    {
        "bootstrap": [True],
        "max_depth": np.arange(1, 20),
        "max_features": np.arange(1, 8),
        "min_samples_leaf": np.arange(1, 10),
        "min_samples_split": np.arange(2, 20),
        "n_estimators": np.arange(50, 150, 10),
    },
    # Parametri XGB
    {
        "n_estimators": np.arange(50, 150, 10),
        "criterion": ["gini", "entropty"],
        "max_depth": np.arange(1, 15),
        "max_features": ["int", "float", "auto", "log2"],
    },
]

# Lista di parametri per il Grid Search
grid_parameters = [
    # Parametri RandomForest
    {
        "bootstrap": [True],
        "max_depth": np.arange(17, 21),
        "max_features": np.arange(5, 9),
        "min_samples_leaf": np.arange(1, 5),
        "min_samples_split": np.arange(15, 19),
        "n_estimators": np.arange(100, 121, 10),
    },
    #  Parametri XGB
    {
        "n_estimators": np.arange(90, 111, 10),
        "criterion": ["gini", "entropty"],
        "max_depth": np.arange(5, 9),
        "max_features": ["log2", "auto"],
    },
]

# Funzione che svolge Random Search
def Random(i, X_train, y_train, cv):
    with open("./FineTuningResults/" + model_name[i] + ".txt", "w") as f:
        random = RandomizedSearchCV(
            models[i],
            random_parameters[i],
            cv=cv,
            random_state=RANDOM_STATE,
            n_iter=100,
            n_jobs=-1,
            scoring="f1_micro",
        )
        result = random.fit(X_train.values, y_train)
        print("Best Score: %s" % result.best_score_)
        print("Best Hyperparameters: %s" % result.best_params_)
        f.write("Random Search")
        f.write(str(models[i]) + str(result.best_params_))
        f.write(str(result.best_score_))


# Funzione che svolge il Grid Search
def Grid(i, X_train, y_train, cv):
    with open("./FineTuningResults/" + model_name[i] + ".txt", "a") as f:
        clf = GridSearchCV(models[i], grid_parameters[i], cv=cv, n_jobs=-1, scoring="f1_micro", verbose=True)
        clf.fit(X_train.values, y_train)
        bestParams = str(clf.best_params_)
        score = str(clf.score(X_train.values, y_train))
        print("\n" + str(models[i]) + " score is:")
        print(score)
        print(bestParams)
        f.write("Grid Search")
        f.write(str(models[i]) + bestParams)
        f.write(score)


# Funzione che si occupa del ciclo e del richiamo delle funzioni di random e grid
def modelTuning(X_train, y_train):
    y_train = y_train.values.ravel()

    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=RANDOM_STATE)

    for i in range(0, len(models)):
        Random(i, X_train, y_train, cv)
        # Grid(i, X_train, y_train, cv)


def main():
    X_train = pd.read_csv("./Datasets/TrainData.csv")
    y_train = pd.read_csv("./Datasets/TrainLabel.csv")

    print(str(models[0]))

    modelTuning(X_train, y_train)


if __name__ == "__main__":
    main()
