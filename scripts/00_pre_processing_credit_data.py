"""
Tratamento de dados da base credit_data.csv
- Valores incoerentes e faltantes substituídos pela média
- Dados padronizados para posterior uso em algoritmos de machine learning
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

filepath = Path('../bases/credit_data.csv')

if __name__ == '__main__':
    base_credit = pd.read_csv(filepath)

    # Descoberta dos dados
    """
    print(base_credit.head(10))
    print(base_credit.tail(8))
    print(base_credit.describe())
    print(base_credit[base_credit['income'] >= 69995.685578])
    print(base_credit[base_credit['loan'] <= 1.377630])
    """
    # Visualização dos dados
    """
    print(np.unique(base_credit['default'], return_counts=True))

    sns.countplot(x=base_credit['default'])

    plt.figure()
    plt.hist(x=base_credit['age'], rwidth=0.95)

    plt.figure()
    plt.hist(x=base_credit['income'], rwidth=0.95)

    plt.figure()
    plt.hist(x=base_credit['loan'], rwidth=0.95)

    grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
    grafico.show()
    plt.show()
    """

    # Tratamento de valores inconsistentes e Faltantes
    """
    print(base_credit.loc[base_credit['age'] < 0])
    print(base_credit[base_credit['age'] < 0])

    # Apagar a coluna inteira
    base_credit2 = base_credit.drop('age', axis = 1)
    print(base_credit2)

    # Apagar os registros com valores faltantes
    base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
    print(base_credit3)
    """

    # Preencher os valores inconsistentes manualmente
    # print(base_credit['age'][base_credit['age'] > 0].mean())
    base_credit.loc[base_credit['age'] < 0, 'age'] = base_credit['age'][base_credit['age'] > 0].mean()

    # Tratamento de valores faltantes
    # print(base_credit.isnull().sum())
    # print(base_credit.loc[pd.isnull(base_credit['age'])])
    base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
    # print(base_credit.isnull().sum())
    # rint(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

    # Divisão entre previsores e classes
    X_credit = base_credit.iloc[:, 1:4].values
    #print(X_credit)
    #print(type(X_credit))

    Y_credit = base_credit.iloc[:, 4].values

    # Escalonamento dos Valores
    print("Income")
    print(X_credit[:, 0].min(), X_credit[:, 0].max())
    print("Age")
    print(X_credit[:, 1].min(), X_credit[:, 1].max())
    print("Loan")
    print(X_credit[:, 2].min(), X_credit[:, 2].max())

    # Padronização é mais indicada quando há outliers na base de dados

    scaler_credit = StandardScaler()
    X_credit = scaler_credit.fit_transform(X_credit)

    print("Income")
    print(X_credit[:, 0].min(), X_credit[:, 0].max())
    print("Age")
    print(X_credit[:, 1].min(), X_credit[:, 1].max())
    print("Loan")
    print(X_credit[:, 2].min(), X_credit[:, 2].max())