from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

if __name__ == '__main__':

    with open('credit.pkl', mode='rb') as f:
        X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

    random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    random_forest_credit.fit(X_credit_treinamento, Y_credit_treinamento)

    previsoes = random_forest_credit.predict(X_credit_teste)


    accuracy = accuracy_score(Y_credit_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print(accuracy)
    print(classification_report(Y_credit_teste, previsoes))


    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(random_forest_credit)
    cm.fit(X_credit_treinamento, Y_credit_treinamento)
    cm.score(X_credit_teste, Y_credit_teste)
    cm.show()