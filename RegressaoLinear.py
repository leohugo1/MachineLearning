import math
import numpy as np




class l1:
    def __init__(self,alfa):
        self.alfa=alfa
    def __call__(self,w):
        return self.alfa * np.linalg.norm(w)
    def grad(self,w):
        return self.alfa * np.sign(w)

class l2:
    def __init__(self,alfa):
        self.alfa=alfa

    def __call__(self,w):
        return self.alfa * 0.5 * w.T.dot(w)
    def grad(self,w):
        return self.alfa * w

class l1_l2:
    def __init__(self,alfa,l1_ratio=0.5):
        self.alfa=alfa
        self.l1_ratio=l1_ratio

    def __call__(self,w):
        l1=self.l1_ratio * np.linalg.norm(w)
        l2=(1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alfa * (l1 + l2)
    def grad(self,w):
        l1=self.l1_ratio * np.sign(w)
        l2=(1 - self.l1_ratio) * w
        return self.alfa * (l1 + l2)

class regressao(object):
    def __init__(self,iteracoes,taxa_aprendizado):
        self.iteracoes=iteracoes
        self.taxa_aprendizado=taxa_aprendizado
    def inicializar_pesos(self,entradas):
        limite=1 / math.sqrt(entradas)
        self.w=np.random.uniform(-limite,limite,(1,entradas))
    def treino(self,x,y):
        x=np.insert(x,0,1,axis=1)
        self.n=x.shape[1]
        self.treino_erros=[]
        self.inicializar_pesos(entradas=self.n)

        for i in range(self.iteracoes):
            y_prev=x.dot(self.w.T)
            mse=np.mean(0.5 * (y - y_prev)**2)
            error = (1/x.shape[0])*np.sum(np.abs(y_prev - y))
            self.treino_erros.append(mse)
            grad_w=-np.dot((y - y_prev).T,x)
            self.w -= self.taxa_aprendizado * grad_w
            if i%(self.iteracoes/10)==0:
                print("porcent: ",(1-error) * 100,"%")
        return self.treino_erros,y_prev

    def previsao(self,x):
        x=np.insert(x,0,1,axis=1)
        y_prev=x.dot(self.w.T)
        return y_prev

class regressao_linear(regressao):
    def __init__(self, iteracoes, taxa_aprendizado,gradient=True):
        self.gradient=gradient
        

        super(regressao_linear,self).__init__(iteracoes=iteracoes, taxa_aprendizado=taxa_aprendizado)
