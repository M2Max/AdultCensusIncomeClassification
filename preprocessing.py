import matplotlib
import pandas as pd
import numpy as np

# Libreria che permette di automatizzare il processo di profiling di un dataset pandas
import pandas_profiling as pp
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, PowerTransformer, MinMaxScaler
from scipy.stats import zscore

# Funzione di over sampling fornita dalla libreria imblearn
from imblearn.over_sampling import SMOTE


# Libreria utilizzata per gestire i warning (nel mio caso per rimuovere quelli fastidiosi e inutili)
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("mode.chained_assignment", None)


from MLFineTuning import RANDOM_STATE

FEATURE_THRESHOLD = 0.04
TEST_SIZE = 0.3


# Funzione che stampa a console alcune informazioni sul dataset
def printInfo(df):
    print("\nDATASET'S HEAD AND TAIL\b")
    print(df.head(10))
    print(df.tail(10))
    print("\nINFORMATION ABOUT THE DATASET STRUCUCTURE \n")
    print(df.info())  # Datatypes and Null values
    print("\nINFORMATION ABOUT THE DATASET'S NUMERICAL FEATURES \n")
    print(df.describe())  # Mean, Std and some other parameters of the dataset's numerical features


# Funzione che si occupa del cleaning del dataset, rimuovendo i duplicati e sostituendo i "?" con il valore della moda di ogni feature
def dataCleaning(df):

    df = df.drop_duplicates(keep="first")

    df["workclass"] = df["workclass"].replace("?", df["workclass"].mode()[0])
    df["occupation"] = df["occupation"].replace("?", df["occupation"].mode()[0])
    df["native.country"] = df["native.country"].replace("?", df["native.country"].mode()[0])

    return df


# Funzione che si occupa del feature engineering appunto
def featureEngineering(df):

    l = LabelEncoder()
    o = OrdinalEncoder()

    # Sostituzione dei valori di Marital.status
    df["marital.status"] = df["marital.status"].replace(
        ["Never-married", "Divorced", "Separated", "Widowed"], "Single"
    )
    df["marital.status"] = df["marital.status"].replace(
        ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"], "Married"
    )
    df["marital.status"] = df["marital.status"].map({"Married": 1, "Single": 0})
    df["marital.status"] = df["marital.status"].astype(int)

    # Utilizzo OrdinalEncoder per le feature categoriche
    for i in df.columns:
        if df[i].dtypes == "O" and i != "income":
            df[i] = o.fit_transform(df[i].values.reshape(-1, 1))

    # Utilizzo LabelEncoder appunto per il label
    df["income"] = l.fit_transform(df["income"])

    # Funzione che stampa la perdita di dati percentuale con i vari treshold di z-score
    def threshold():
        for i in np.arange(3, 5, 0.2):
            data = df.copy()
            data = data[(z < i).all(axis=1)]
            loss = (df.shape[0] - data.shape[0]) / df.shape[0] * 100
            print("With threshold {} data loss is {}%".format(np.round(i, 1), np.round(loss, 2)))

    z = np.abs(zscore(df))
    print("INFORMATION ABOUT Z-SCORE AND CORRELATED DATA LOSS")
    print(threshold())

    # Rimuovo tutte le istanze con z-score maggiore di 4.2
    df = df[(z < 4.2).all(axis=1)]

    pt = PowerTransformer()

    numerical_features = [
        "education.num",
        "age",
        "fnlwgt",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]

    # Applico PowerTransformer per ridurre l'assimetria (Skewness) dei dati
    for i in numerical_features:
        if np.abs(df[i].skew()) > 0.5:
            df[i] = pt.fit_transform(df[i].values.reshape(-1, 1))

    over = SMOTE()

    y = df.pop("income")
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Effettuo l'over sampling con l'algoritmo SMOTE
    X_train, y_train = over.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


# Funzione che si occupa di feature selection, restituendo X, y già private delle feature con mutua informazione sotto il treshold definito
def featureSelection(X_train, X_test, y_train):
    bestfeatures = SelectKBest(score_func=mutual_info_classif, k=2)
    fit = bestfeatures.fit(X_train, y_train)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]
    print(featureScores)

    features = (
        featureScores[featureScores["Score"] < FEATURE_THRESHOLD]["Specs"]
        .reset_index(drop=True)
        .to_numpy()
    )
    for f in features:
        X_train = X_train.drop(f, axis=1)
        X_test = X_test.drop(f, axis=1)

    return X_train, X_test


# Funzione che si occupa feature scaling, calcola i valori per lo scaling su X_train
# e poi li utilizza sia per X_train che per X_test
def featureScaling(X_train, X_test):

    print(X_train.head(10))
    print(X_train.tail(10))

    # Feature Scaling using Min Max approach
    scaler = MinMaxScaler()
    xd = scaler.fit_transform(X_train)
    xd1 = scaler.transform(X_test)
    X_train = pd.DataFrame(xd, columns=X_train.columns)
    X_test = pd.DataFrame(xd1, columns=X_test.columns)

    print(X_train.head(10))
    print(X_train.tail(10))

    return X_train, X_test


def main():
    df = pd.read_csv("./Datasets/Adult.csv")
    # Crea il file reportBefore.html che permette facilmente di comprendere, anche abbastanza approfonditamente, il dataset nella sua versione originale
    #pp.ProfileReport(df, title="Census Income Profiling Report", explorative=True).to_file(
    #    "reportBefore.html"
    #)
    matplotlib.use("tkagg")

    printInfo(df)
    df = dataCleaning(df)
    X_train, X_test, y_train, y_test = featureEngineering(df)
    X_train, X_test = featureSelection(X_train, X_test, y_train)

    X_train, X_test = featureScaling(X_train, X_test)

    X_train.to_csv("./Datasets/TrainData.csv", index=False)
    X_test.to_csv("./Datasets/TestData.csv", index=False)
    y_train.to_csv("./Datasets/TrainLabel.csv", index=False)
    y_test.to_csv("./Datasets/TestLabel.csv", index=False)


if __name__ == "__main__":
    main()
