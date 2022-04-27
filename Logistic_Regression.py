
import numpy as np
import pandas as pd





def modelo(x,y,aprend,iteracao):
    m=x.shape[1]
    n=x.shape[0]
    w=np.zeros((n,1))
    B=0
    error_list=[]

    for i in range(iteracao):
        y_pred = np.dot(w.T,x) + B
        A=1/(1+ np.exp(-y_pred))
        cost=-(1/m)*np.sum(y*np.log(A) + (1-y)*np.log(1-A))
        dw=(1/m)*np.dot(A-y,x.T)
        db=(1/m)*np.sum(A-y)

        w=w - aprend * dw.T
        B=B - aprend *db
        error_list.append(cost)
        if i%(iteracao/10)==0:
            print("cost after :",i,"iteration is: ",cost)
    return w,B,error_list










iteracao=1000
aprend=0.0015


X_train = pd.read_csv("train_X.csv")
Y_train = pd.read_csv("train_Y.csv")

X_test = pd.read_csv("test_X.csv")
Y_test = pd.read_csv("test_Y.csv")
X_train = X_train.drop("Id", axis = 1)
Y_train = Y_train.drop("Id", axis = 1)
X_test = X_test.drop("Id", axis = 1)
Y_test = Y_test.drop("Id", axis = 1)
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])

X_test = X_test.T
Y_test = Y_test.reshape(1, X_test.shape[1])
w,b,error_list=modelo(X_train,Y_train,aprend,iteracao)
