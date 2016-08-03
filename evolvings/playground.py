from repo.epl_krls import ePLKRLSRegressor
from repo.epl import ePLRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error

raw_data = np.loadtxt("data/ibov.txt")
nrm_data = MinMaxScaler((0 ,1)).fit_transform(raw_data)

# Generating data for the first step-ahead forecast
X = [nrm_data[i:i+2] for i in range(len(nrm_data) - 2)]
y = [nrm_data[i+2] for i in range(len(nrm_data) - 2)]

epl = ePLRegressor()
eplk = ePLKRLSRegressor()

for i in range(1, len(nrm_data) - 2):
	eplk.evolve(X[i], y[i-1])
eplk.plot_centers(X[1:])