
from Logistic_Regression import RegressaoLogistica
from RegressaoLinear import *
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("admicao.csv")
test=pd.read_csv("housing_test.csv")
train_data = train.values
test_data=test.values




Y = train_data[:, -1].reshape(train_data.shape[0], 1)
X = train_data[:, :-1]
#X_test=test_data[:, :-1]
#y_test=test_data[:, -1].reshape(test_data.shape[0], 1)
print(Y.shape)



regressao=RegressaoLogistica(0.001,1000,True)
y_prev1,error_list=regressao.treino(X,Y)

y_prev=regressao.previsao(x=X)

print(y_prev)


