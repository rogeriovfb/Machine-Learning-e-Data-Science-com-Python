from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

if __name__ == '__main__':

    with open('census.pkl', mode='rb') as f:
        X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

    naive_census = GaussianNB()
    naive_census.fit(X_census_treinamento, Y_census_treinamento)

    previsoes = naive_census.predict(X_census_teste)

    print(accuracy_score(Y_census_teste, previsoes))
    print(classification_report(Y_census_teste, previsoes))

    # Aleatório seria mais eficaz que o algoritmo, pois existem apenas duas classes (50% no aleatório)
    # Se não executar o escalonamento, algoritmo chega próximo de 70%

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(naive_census)
    cm.fit(X_census_treinamento, Y_census_treinamento)
    cm.score(X_census_teste, Y_census_teste)
    cm.show()
