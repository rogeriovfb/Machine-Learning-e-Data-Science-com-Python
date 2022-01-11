from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

from sklearn import tree
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    with open('census.pkl', mode='rb') as f:
        X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

    arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arvore_census.fit(X_census_treinamento, Y_census_treinamento)

    previsoes = arvore_census.predict(X_census_teste)

    accuracy = accuracy_score(Y_census_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print(accuracy)
    print(classification_report(Y_census_teste, previsoes))

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(arvore_census)
    cm.fit(X_census_treinamento, Y_census_treinamento)
    cm.score(X_census_teste, Y_census_teste)
    cm.show()
