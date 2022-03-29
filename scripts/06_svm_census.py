from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

if __name__ == '__main__':

    with open('census.pkl', mode='rb') as f:
        X_census_treinamento, Y_census_treinamento, X_census_teste, Y_census_teste = pickle.load(f)

    svm_census = SVC(kernel='linear', random_state=1, C=1.0)
    svm_census.fit(X_census_treinamento, Y_census_treinamento)

    previsoes = svm_census.predict(X_census_teste)

    accuracy = accuracy_score(Y_census_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print("Accuracy = " + str(accuracy))
    print(classification_report(Y_census_teste, previsoes))

    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(svm_census)
    cm.fit(X_census_treinamento, Y_census_treinamento)
    cm.score(X_census_teste, Y_census_teste)
    cm.show()
