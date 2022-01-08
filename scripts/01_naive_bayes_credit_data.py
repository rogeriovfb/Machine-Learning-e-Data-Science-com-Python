from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

if __name__ == '__main__':

    with open('credit.pkl', mode='rb') as f:
        X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

    naive_credit_data = GaussianNB()
    naive_credit_data.fit(X_credit_treinamento, Y_credit_treinamento)

    previsoes = naive_credit_data.predict(X_credit_teste)

    print(accuracy_score(Y_credit_teste, previsoes))

    # Acertos e erros de clientes que pagam e não pagam
    print(confusion_matrix(Y_credit_teste, previsoes))
    print(classification_report(Y_credit_teste, previsoes))

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(naive_credit_data)
    cm.fit(X_credit_treinamento, Y_credit_treinamento)
    cm.score(X_credit_teste, Y_credit_teste)
    cm.show()




