from repo.epl import ePLRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error

epl = ePLRegressor()

X = MinMaxScaler((0 ,1)).fit_transform(np.array([[i, i+1] for i in range(100)]))
y = MinMaxScaler((0 ,1)).fit_transform(np.array([i for i in range(100)]))

res = []
for i in range(1, 100):
	res.append(epl.evolve(X[i], y[i-1]))

print mean_squared_error(y[1:], np.array(res))