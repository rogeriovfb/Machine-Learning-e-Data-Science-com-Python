from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

if __name__ == '__main__':

    with open('credit.pkl', mode='rb') as f:
        X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

    knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn_credit.fit(X_credit_treinamento, Y_credit_treinamento)

    previsoes = knn_credit.predict(X_credit_teste)
    accuracy = accuracy_score(Y_credit_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print(accuracy)
    print(classification_report(Y_credit_teste, previsoes))

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(knn_credit)
    cm.fit(X_credit_treinamento, Y_credit_treinamento)
    cm.score(X_credit_teste, Y_credit_teste)
    cm.show()