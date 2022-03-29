from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

if __name__ == '__main__':

    with open('credit.pkl', mode='rb') as f:
        X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

    svm_credit = SVC(kernel='rbf', random_state=1, C=2.0)
    svm_credit.fit(X_credit_treinamento, Y_credit_treinamento)

    previsoes = svm_credit.predict(X_credit_teste)

    accuracy = accuracy_score(Y_credit_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print("Accuracy = " + str(accuracy))
    print(classification_report(Y_credit_teste, previsoes))

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(svm_credit)
    cm.fit(X_credit_treinamento, Y_credit_treinamento)
    cm.score(X_credit_teste, Y_credit_teste)
    cm.show()
