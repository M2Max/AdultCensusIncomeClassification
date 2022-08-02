from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import keras

from scikeras.wrappers import KerasClassifier

import os

from MLFineTuning import RANDOM_STATE

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"  # Elimina i warning di Tensorflow sull'assenza di CUDA core

X_train = pd.read_csv("./Datasets/TrainData.csv")
y_train = pd.read_csv("./Datasets/TrainLabel.csv")

# Funzione che crea il modello Keras con i parametri passati
def create_model(optimizer="adam"):
    model = models.Sequential()
    model.add(layers.Dense(12, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(9, activation="relu"))
    model.add(layers.Dense(6, activation="relu"))
    model.add(layers.Dense(3, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[keras.metrics.Precision()])

    return model


def main():
    # Wrapper del modello keras che lo rende compatibile con le funzioni di sklearn
    model = KerasClassifier(model=create_model)

    # Parametri per il random search
    random_params = {
        "epochs": np.arange(10, 21),
        "batch_size": np.arange(1, 20),
        "optimizer": ["SGD", "Adadelta", "Adam", "Adamax", "Nadam"],
        "optimizer__learning_rate": np.linspace(0.001, 1, 10),
    }

    # Parametri per il grid search (creati dopo una run del random search e basati sul risultato)
    grid_params = {
        "epochs": np.arange(14, 18),
        "batch_size": np.arange(1, 9),
        "optimizer": ["SGD"],
        "optimizer__learning_rate": np.linspace(0.05, 0.2, 5),
    }

    # Oggetto necessario per la cross validation
    cv = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE, shuffle=True)
    with open("./FineTuningResults/KerasRandom.txt", "w") as f:
        random = RandomizedSearchCV(
            model,
            random_params,
            scoring="f1",
            cv=cv,
            n_iter=100,
            n_jobs=-1,
        )
        result = random.fit(X_train.values, y_train)
        print("Best Score: %s" % result.best_score_)
        print("Best Hyperparameters: %s" % result.best_params_)
        f.write("Random Search")
        f.write(str(model) + str(result.best_params_))
        f.write(str(result.best_score_))


    #Codice commentato che effettuerebbe GridSearchCV sul modello Keras
    
    # with open("./FineTuningResults/KerasGrid.txt", "a") as f:
    #     grid = GridSearchCV(model, grid_params, scoring="f1", n_jobs=10, cv=cv)
    #     grid_result = grid.fit(X_train, y_train)
    #     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #     means = grid_result.cv_results_["mean_test_score"]
    #     stds = grid_result.cv_results_["std_test_score"]
    #     params = grid_result.cv_results_["params"]
    #     for mean, stdev, param in zip(means, stds, params):
    #         print("%f (%f) with: %r" % (mean, stdev, param))
    #     f.write("Grid Search")
    #     f.write(str(model) + str(grid_result.best_params_))
    #     f.write(str(grid_result.best_score_))


if __name__ == "__main__":
    main()
