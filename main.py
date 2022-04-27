
from RegressaoLinear import *
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("housing.csv")
test=pd.read_csv("housing_test.csv")
train_data = train.values
test_data=test.values

Y = train_data[:, -1].reshape(train_data.shape[0], 1)
X = train_data[:, :-1]
X_test=test_data[:, :-1]
y_test=test_data[:, -1].reshape(test_data.shape[0], 1)




regressao=regressao_linear(1000,0.000001,True)
trino_error,y_prev1=regressao.treino(X,Y)
#print(trino_error)
y_prev=regressao.previsao(x=X_test)


plt.figure(figsize=(20,5))
plt.plot(y_prev1, linewidth=2, color='r')
plt.plot(Y, linewidth=0.5,color='b')
plt.title('Valores preditos e os valores reais',size=15)
plt.legend(['Predições','Real'],fontsize=15)
plt.show()


