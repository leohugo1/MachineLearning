
import math
import numpy as np




class RegressaoLogistica:
    def __init__(self,taxa_aprendizado,iteracoes,gradient=True):
        self.taxa_aprendizado=taxa_aprendizado
        self.gradient=gradient
        self.iteracoes=iteracoes
    
    def inicializar_pesos(self,x):
        entradas=x.shape[1]
        limite=1/math.sqrt(entradas)
        self.w=np.random.uniform(-limite,limite,(1,entradas))
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def treino(self,x,y):
        n=len(x)
        self.inicializar_pesos(x)
        self.erro_list=[]
        for i in range(self.iteracoes):
            y_prev=x.dot(self.w.T)
            sigmoid=self.sigmoid(y_prev)
            e1=np.multiply(-y,np.log(sigmoid))
            e2=np.multiply((1-y),np.log(1-sigmoid))
            erro=np.sum(e1-e2)
            error=erro/n
            self.erro_list.append(error)
            w_prev= -np.dot((y - sigmoid).T,x)
            self.w += self.taxa_aprendizado * w_prev
        return sigmoid,self.erro_list
    def previsao(self,x):
        y_pred=x.dot(self.w.T)
        sigmoid=(self.sigmoid(y_pred)) >= 0.5
        return (sigmoid.astype(int))