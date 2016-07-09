import numpy as np


class RLS(object):
    
    def __init__(self, w, gama):
        self.__w = w
        self.gama = gama
        self.B = np.mat(np.eye(w.size))

    def update(self, X, y):
        mxt = np.mat(X).T
        A = self.B/(self.gama + mxt.T*self.B*mxt)
        self.__w = self.__w +  (y - np.dot(self.__w, X))*np.array(A*mxt).T
        self.B = self.gama**-1 * (np.eye(self.__w.size) - A*mxt*mxt.T)*self.B
        
        return self

    def getw(self):
        return self.__w

    def setw(self, w):
        self.__w = w