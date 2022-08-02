# AdultCensusIncomeClassification

Nella cartella Images sono presenti tutte le immagini utilizzate nella relazione in qualità superiore

Nella cartella FineTuningResults sono presenti i risultati scritti dal programma dei finetuning dei vari modelli

Nella cartella datasets i vari file csv, quello iniziale e quelli derivati utilizzati nei vari script

La relazione è il file RelazioneProgetto.pdf

Eseguendo DataVisualization.py vengono aggiornati tutti i graifici delle feature (bar plot, pie chart, boxplot, histogram)
Se invece si esegue preprocessing.py vengono ricreati i dataset eseguendo tutte le operazioni di preprocessing sul dataset iniziale
StratifiedKFoldCrossValidation.py stampa a schermo i risultati di una RepeatedStratifiedKFoldCrossValidation su tutti i modelli di ML 
Rispettivamente MLFineTuning.py e IAFineTuning.py effettuano randomized e grid search su i modelli di ML e la rete neurale sequenziale Keras
RandomForest.py esegue training e test del modello definitivo RandomForest, XGBoost.py fa la stessa cosa per XGBoost
Infine Confronto.py allena e testa entrambi i modelli, prima sul dataset iniziale e poi su quello preprocessato
