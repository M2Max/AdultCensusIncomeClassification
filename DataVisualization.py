import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling as pp
import seaborn as sns

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Data visualization utilizzando un grafico a torta e un grafico a barre
# per visualizzare la frequenza dei valori di "income"
def genericVisualization(df):

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.title(label="Income frequency visualization - Pie Chart")
    df["income"].value_counts().plot.pie(autopct="%1.1f%%")

    plt.subplot(1, 2, 2)
    plt.title(label="Income frequency visualization - Bar Chart")
    sns.countplot(x="income", data=df)
    plt.ylabel("No. of People")
    df["income"].value_counts()
    plt.savefig("./Images/Before/generic.png", bbox_inches="tight")


# Data visualization utilizzando boxplots per le alcune feature numeriche
def outliersVisualization(df):

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for column, ax in zip(df.select_dtypes(include="number").columns, axes.flat):
        sns.boxplot(df[column], ax=ax)
        ax.set_title(label=column + " Boxplot")

    plt.savefig("./Images/Before/boxplots.png", bbox_inches="tight")


# Data visualization utilizzando grafici a barre per ogni feature categorica
def visualizeCategorical(df):

    categorical_features = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "sex",
        "race",
    ]

    for feature in categorical_features:
        if feature == "occupation" or feature == "marital.status" or feature == "education":  #
            figure = plt.figure(figsize=(18, 6))
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99)
        else:
            figure = plt.figure(figsize=(11, 6))
            plt.subplots_adjust(left=0.08, bottom=0.1, right=0.99, top=0.99)
        sns.countplot(df[feature], hue=df["income"])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("./Images/Before/" + feature + ".png", bbox_inches="tight")


# Data visualization utilizzando istogrammi per ogni feature numerica
def visualizeNumerical(df):

    numerical_features = [
        "education.num",
        "age",
        "fnlwgt",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]

    for feature in numerical_features:
        figure = plt.figure(figsize=(12, 6))
        sns.histplot(df, x=feature, hue="income", stat="frequency")
        plt.savefig("./Images/Before/" + feature + ".png", bbox_inches="tight")


# Richiama tutte le funzioni di data visualization, tutti i grafici vengono salvati in nella directory ./Images/Before/
def main():
    df = pd.read_csv("./Datasets/Adult.csv")

    matplotlib.use("tkagg")
    sns.set()
    sns.color_palette("bright")

    visualizeCategorical(df)
    visualizeNumerical(df)
    genericVisualization(df)
    outliersVisualization(df)


if __name__ == "__main__":
    main()
