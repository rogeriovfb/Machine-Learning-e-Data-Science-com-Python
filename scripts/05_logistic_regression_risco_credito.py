from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

if __name__ == '__main__':

    with open('risco_credito.pkl', mode='rb') as f:
        X_risco_credito, Y_risco_credito = pickle.load(f)

    # Apagar risco moderado da base
    X_risco_credito = np.delete(X_risco_credito, [2, 7, 11], axis=0)
    Y_risco_credito = np.delete(Y_risco_credito, [2, 7, 11], axis=0)

    logistic_risco_credito = LogisticRegression(random_state=1)
    logistic_risco_credito.fit(X_risco_credito, Y_risco_credito)
    print("B0 = " + str(logistic_risco_credito.intercept_))
    print("B1,2,3,4 = " + str(logistic_risco_credito.coef_))

    #História de crédito = Boa(0), Dívida = Alta (0), Garantias = Nenhuma (1), Renda > 35 (2)
    #História de crédito = Ruim (2), Dívida = Alta(0), Garantias = Adequada (0), Renda < 15 (0)
    previsoes =logistic_risco_credito.predict([[0,0,1,2], [2,0,0,0]])

    print(previsoes)