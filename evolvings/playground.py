from repo.epl_krls import ePLKRLSRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error

imin, imax = np.loadtxt("data/bench_hourly"), np.loadtxt("data/bench_daily")
nmin, nmax = MinMaxScaler((0 ,1)).fit_transform(imin), \
             MinMaxScaler((0 ,1)).fit_transform(imax)

imm = 0
for idata, ndata in zip([imin, imax], [nmin, nmax]):
    imm = imm + 1
    # Generating ndata for the first step-ahead forecast
    X = [ndata[i:i+2] for i in range(len(ndata) - 2)]
    y = [ndata[i+2] for i in range(len(ndata) - 2)]

    eplk = ePLKRLSRegressor()

    nres = []
    for i in range(1, len(ndata) - 2):
        r = eplk.evolve(X[i], y[i-1])
        nres.append(r)

    ires = np.array([int(min(idata) + (max(idata) - min(idata)) * i) for i in nres])
    with open(str(imm) + ".txt", 'w+') as w:
        for ir in ires:
            w.write(str(ir) + "\n")

    np.savetxt(str(imm) + ".txt", ires)
    #print np.sqrt(mean_squared_error([idata[i+1] for i in range(len(ndata) - 1)][:-1], ires))
    #eplk.plot_centers(X[1:])