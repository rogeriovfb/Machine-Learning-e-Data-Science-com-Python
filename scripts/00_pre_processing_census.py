"""
Tratamento de dados da base census.csv

"""
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

filepath = Path('../bases/census.csv')
if __name__ == '__main__':
    base_census = pd.read_csv(filepath)

    # Descoberta dos dados
    """
    print(base_census)
    pd.set_option('display.max_columns', 15)
    print(base_census.describe())
    print(base_census.isnull().sum())
    """
    # Visualização dos dados
    """
    print(np.unique(base_census['income'], return_counts=True))

    sns.countplot(x=base_census['income'])

    plt.figure()
    plt.hist(x=base_census['age'], rwidth=0.95)

    plt.figure()
    plt.hist(x=base_census['education-num'], rwidth=0.95)

    plt.figure()
    plt.hist(x=base_census['hour-per-week'], rwidth=0.95)

    grafico = px.treemap(base_census, path=['workclass', 'age'])
    #grafico.show()

    grafico = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
    #grafico.show()

    grafico = px.parallel_categories(base_census, dimensions=['education', 'occupation', 'income'])
    #grafico.show()

    grafico = px.parallel_categories(base_census, dimensions=['education', 'income'])
    grafico.show()

    #plt.show()
    """
    # Divisão entre previsores e classes
    X_census = base_census.iloc[:, 0:14].values
    Y_census = base_census.iloc[:, 14].values

    # Tratamento de atributos categóricos
    columns_cat = [1, 3, 5, 6, 7, 8, 9, 13]
    # LabelEncoder
    label_encoder = []
    for i in columns_cat:
        label_encoder.append(LabelEncoder())
        X_census[:, i] = label_encoder[-1].fit_transform(X_census[:, i])
    #print(X_census)

    # OneHotEncoder
    onehotencoder_census = ColumnTransformer(
        transformers=[('OneHot', OneHotEncoder(), columns_cat)], remainder='passthrough')
    X_census = onehotencoder_census.fit_transform(X_census).toarray()
    #print(X_census[0])
    #print(X_census.shape)

    # Escalonamento dos Valores
    scaler_census = StandardScaler()
    X_census = scaler_census.fit_transform(X_census)
    #print(X_census[0])

    # Divisão das Bases em treinamento e teste
    X_census_treinamento, X_census_teste, Y_census_treinamento, Y_census_teste = \
        train_test_split(X_census, Y_census, test_size=0.15, random_state=0)

    with open('census.pkl', mode='wb') as f:
        pickle.dump([X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste], f)