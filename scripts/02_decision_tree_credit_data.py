from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

from sklearn import tree
import matplotlib.pyplot as plt

import pickle

if __name__ == '__main__':

    with open('credit.pkl', mode='rb') as f:
        X_credit_treinamento, Y_credit_treinamento, X_credit_teste, Y_credit_teste = pickle.load(f)

    arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
    arvore_credit.fit(X_credit_treinamento, Y_credit_treinamento)

    previsoes = arvore_credit.predict(X_credit_teste)

    accuracy = accuracy_score(Y_credit_teste, previsoes)

    # Acertos e erros de clientes que pagam e não pagam
    print(accuracy)
    print(classification_report(Y_credit_teste, previsoes))


    # Confusion Matrix usando yellowbrick
    cm = ConfusionMatrix(arvore_credit)
    cm.fit(X_credit_treinamento, Y_credit_treinamento)
    cm.score(X_credit_teste, Y_credit_teste)
    #cm.show()

    # Arvore
    previsores = ['income', 'age', 'loan']
    figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    tree.plot_tree(arvore_credit, feature_names=previsores, class_names=['0', '1'], filled=True)
    plt.show()

