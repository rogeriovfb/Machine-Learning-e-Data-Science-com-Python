from sklearn.naive_bayes import GaussianNB
import pickle

if __name__ == '__main__':

    with open('risco_credito.pkl', mode='rb') as f:
        X_risco_credito, Y_risco_credito = pickle.load(f)

    naive_risco_credito = GaussianNB()
    naive_risco_credito.fit(X_risco_credito, Y_risco_credito)

    #Previsão

    #História de crédito = Boa(0), Dívida = Alta (0), Garantias = Nenhuma (1), Renda > 35 (2)
    #História de crédito = Ruim (2), Dívida = Alta(0), Garantias = Adequada (0), Renda < 15 (0)
    previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
    print(previsao)

    # Classes existentes na base
    print(naive_risco_credito.classes_)
    # Número de registros de cada classe
    print(naive_risco_credito.class_count_)
    # Quanto cada classe representa da base de dados (%)
    print(naive_risco_credito.class_prior_)
    