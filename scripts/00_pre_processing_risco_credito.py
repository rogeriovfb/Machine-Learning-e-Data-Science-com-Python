from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

filepath = Path('../bases/risco_credito.csv')

if __name__ == '__main__':
    base_risco_credito = pd.read_csv(filepath)

    # Descoberta dos dados
    #print(base_risco_credito)

    # Divisão entre previsores e classes
    X_risco_credito = base_risco_credito.iloc[:, 0:4].values
    Y_risco_credito = base_risco_credito.iloc[:, 4].values

    # Tratamento de atributos categóricos
    columns_cat = [0, 1, 2, 3] #historia, divida, garantia, renda
    # LabelEncoder
    label_encoder = []
    for i in columns_cat:
        label_encoder.append(LabelEncoder())
        X_risco_credito[:, i] = label_encoder[-1].fit_transform(X_risco_credito[:, i])

    with open('risco_credito.pkl', mode='wb') as f:
        pickle.dump([X_risco_credito, Y_risco_credito], f)
