import numpy as np
import matplotlib.pyplot as plt
from repo.epl_krls import ePLKRLSRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.mlab as mlab
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from random import normalvariate
from handler import Handler


def arrange_data(d):
    X = []
    for p in [d[x:x + 24] for x in range(0, len(d), 24)][:-1]:
        # Defining X the same for all 24 hours in day-ahead
        for i in range(24):
            X.append(p)
    return np.array(X), d[24:]


def calibrate_arma(ar=5, ma=1, data=None):
    # Fitting an ARMA model for the required data
    arma = sm.tsa.ARMA(data, (ar, ma)).fit(disp=0)
    # Capturing the ARMA params
    params = arma.params
    # Splitting the params into AR and MA params
    arparams = np.r_[1, -np.array(
        params[len(params) - (ar + ma):len(params) - ma])]
    maparams = np.r_[1, np.array(params[len(params) - ma:])]

    # Creating a callback function for generating new series
    def gen(nsample):
        # Same mean and standard deviation
        return [normalvariate(np.average(arma.resid), np.std(
            arma.resid)) for i in range(nsample)]

    # Generating new time series with the same properties of data
    return arma_generate_sample(
        arparams, maparams, len(data), distrvs=gen)


def save_file(path, data):
    with open('data/pjm/%s' % path, 'w+') as w:
        for d in data:
            w.write(str(int(np.ceil(d))) + "\n")

# Handler class
handler = Handler()

for fig, name in enumerate(['012015', '042015', '072015', '102015']):
    # Initializing KRLS forecasting model
    krls = ePLKRLSRegressor()

    # Initializing training and testing data
    trn_data = np.loadtxt('data/pjm/train/%s' % name)
    tst_data = np.loadtxt('data/pjm/test/%s' % name)

    # Normalizing the two datasets
    norm_trn_data = MinMaxScaler((0, 1)).fit_transform(trn_data)
    norm_tst_data = MinMaxScaler((0, 1)).fit_transform(tst_data)

    # Training the KRLS model
    x_trn, y_trn = arrange_data(norm_trn_data)
    [krls.evolve(
        x_trn[i + 1], y_trn[i]) for i in range(len(x_trn) - 1)]

    # Testing the KRLS model
    x_tst, y_tst = arrange_data(norm_tst_data)
    fcast = [krls.evolve(
        x_tst[i + 1], y_tst[i]) for i in range(len(x_tst) - 1)]

    # Denormalizing the testing data
    fcast_data = np.array([int(min(tst_data) + (
        max(tst_data) - min(tst_data)) * k) for k in fcast])

    ax1 = plt.subplot(3, 4, fig + 1)

    months = ['January', 'April', 'July', 'October']
    figure_title = '%s of 2015' % months[fig]
    ax1.text(0.5, 1.08, figure_title,
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlim(xmax=725, xmin=0)
    ax1.set_ylim(ymax=7000, ymin=-100)

    ax1.set_yticks([0, 1500, 3000, 4500, 6000])

    ax1.plot(
        tst_data[25:], color="blue", label='Observed wind data', linewidth=1.3)
    ax1.plot(
        fcast_data, color="green", label='ePL-KRLS forecast', linewidth=1.3)

    # ---------------------------------------------------------- #

    err_data = tst_data[25:] - fcast_data
    ax2 = plt.subplot(3, 4, fig + 5)
    ax2.set_xlim(xmax=725, xmin=0)
    ax2.set_ylim(ymax=2500, ymin=-2500)
    plt.axhline(0, color='gray', linestyle='--')
    ax2.plot(err_data, color="red", linewidth=1.0, label='Observed')
    ax2.set_yticks([-2000, -1000, 0, 1000, 2000])

    # ---------------------------------------------------------- #

    mu, sigma = np.average(err_data), np.std(err_data)

    # the histogram of the data
    ax3 = plt.subplot(3, 4, fig + 9)
    n, bins, patches = plt.hist(
        err_data, 70, normed=1, facecolor='purple', alpha=0.50)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)

    synt_data = calibrate_arma(data=err_data)
    plt.plot(bins, y, linestyle='--', color='purple', linewidth=1.5, label='Observed')

    mu, sigma = np.average(synt_data), np.std(synt_data)
    n, bins, patches = plt.hist(
        synt_data, 70, normed=1, facecolor='#000073', alpha=0.50)

    synd_ts = MinMaxScaler((0, max(
        fcast_data + synt_data))).fit_transform(fcast_data + synt_data)
    ax2.plot(synt_data, color='orange', linewidth=0.7, label='Simulated')
    ax1.plot(synd_ts, color='black', linestyle='--', linewidth=1.2, label='Simulated wind data')

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)

    ax3.plot(bins, y, linestyle='--', color='#000073', linewidth=1.5, label='Simulated with ARMA(5, 1)')

    if fig == 0:
        ax1.set_ylabel('Total Power (in MW)', {'fontsize': 14})
        ax2.set_ylabel('Forecast Error (in MW)', {'fontsize': 14})
        ax3.set_ylabel('Relative Error Frequency (%)', {'fontsize': 14})
    else:
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

    ax3.set_yticks([0.000, 0.0005, 0.001, 0.0015, 0.002])
    plt.yticks([0.000, 0.00075, 0.0015, 0.00225, 0.003],
               ['0.00%', '0.75%', '1.50%', '2.25%', '3.0%'])
    plt.axis([-1500, 1500, 0, 10])
    ax3.set_ylim(ymax=0.003, ymin=0)

    # Inserting legends
    handler.insert_legend(ax1)
    handler.insert_legend(ax2)
    handler.insert_legend(ax3)

    # Saving data
    save_file('fcast/%s' % name, fcast_data)
    save_file('error/%s' % name, err_data)
    save_file('synt/%s' % name, synt_data)

plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.95,
                    wspace=0.05, hspace=0.15)
plt.show()
