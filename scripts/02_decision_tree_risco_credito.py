from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

import pickle

if __name__ == '__main__':

    with open('risco_credito.pkl', mode='rb') as f:
        X_risco_credito, Y_risco_credito = pickle.load(f)

    arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
    arvore_risco_credito.fit(X_risco_credito, Y_risco_credito)

    print(arvore_risco_credito.feature_importances_)
    previsores = ['história', 'dívida', 'garantias', 'renda']
    figura, eixos = plt.subplots(nrows = 1, ncols=1, figsize=(10,10))
    tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_,
                   filled=True)

    # Previsão

    # História de crédito = Boa(0), Dívida = Alta (0), Garantias = Nenhuma (1), Renda > 35 (2)
    # História de crédito = Ruim (2), Dívida = Alta(0), Garantias = Adequada (0),
    previsoes = arvore_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
    print(previsoes)

    plt.show()
