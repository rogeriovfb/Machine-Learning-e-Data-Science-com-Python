from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

if __name__ == '__main__':

    with open('census.pkl', mode='rb') as f:
        X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

    random_forest_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    random_forest_census.fit(X_census_treinamento, Y_census_treinamento)

    previsoes = random_forest_census.predict(X_census_teste)


    accuracy = accuracy_score(Y_census_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print(accuracy)
    print(classification_report(Y_census_teste, previsoes))


    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(random_forest_census)
    cm.fit(X_census_treinamento, Y_census_treinamento)
    cm.score(X_census_teste, Y_census_teste)
    cm.show()