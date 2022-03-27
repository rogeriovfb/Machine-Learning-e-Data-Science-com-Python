from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

if __name__ == '__main__':

    with open('census.pkl', mode='rb') as f:
        X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

    logistic_census = LogisticRegression(random_state=1)
    logistic_census.fit(X_census_treinamento, Y_census_treinamento)

    print("B0 = " + str(logistic_census.intercept_))
    print("B1,2,3 = " + str(logistic_census.coef_))

    previsoes = logistic_census.predict(X_census_teste)
    accuracy = accuracy_score(Y_census_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print("Accuracy = " + str(accuracy))
    print(classification_report(Y_census_teste, previsoes))

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(logistic_census)
    cm.fit(X_census_treinamento, Y_census_treinamento)
    cm.score(X_census_teste, Y_census_teste)
    cm.show()
