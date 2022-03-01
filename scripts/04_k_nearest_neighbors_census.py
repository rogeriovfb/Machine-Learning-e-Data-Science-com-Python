from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

if __name__ == '__main__':

    with open('census.pkl', mode='rb') as f:
        X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

    knn_census =     knn_credit = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
    knn_census.fit(X_census_treinamento, Y_census_treinamento)

    previsoes = knn_census.predict(X_census_teste)


    accuracy = accuracy_score(Y_census_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print(accuracy)
    print(classification_report(Y_census_teste, previsoes))


    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(knn_census)
    cm.fit(X_census_treinamento, Y_census_treinamento)
    cm.score(X_census_teste, Y_census_teste)
    cm.show()