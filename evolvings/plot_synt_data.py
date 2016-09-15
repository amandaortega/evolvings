import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from random import normalvariate
from handler import Handler


def calibrate_arma(ar=3, ma=2, data=None):
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
    return MinMaxScaler((min(data), max(data))).fit_transform(
        arma_generate_sample(arparams, maparams, len(data), distrvs=gen))


# Handler class
handler = Handler()

for fig, name in enumerate(['18042016']):
    # Initializing testing and the forecast data
    tst_data = np.loadtxt('data/ons/test/%s' % name)
    fcast_data = np.loadtxt('data/ons/fcast/%s' % name)
    error_data = tst_data - fcast_data
    ax1 = plt.subplot(1, 1, 1)

    months = ['August 10, 2015', 'November 30, 2015',
              'April 18, 2016', 'June 27, 2016']
    figure_title = '%s' % months[fig]
    ax1.text(0.5, 1.04, figure_title,
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlim(xmax=1450, xmin=-10)
    ax1.set_ylim(ymax=2000, ymin=-2000)

    ax1.set_yticks([-3000, -1500, 0, 1500, 3000])

    mu, sigma = np.average(error_data), np.std(error_data)

    # the histogram of the data
    n, bins, patches = plt.hist(
        error_data, 70, normed=1, alpha=0.00)

    # add a 'best fit' line
    for i in range(4):
        synt_data = calibrate_arma(data=error_data)

        synd_ts = MinMaxScaler((min(fcast_data), max(
            fcast_data + synt_data))).fit_transform(fcast_data + synt_data)

        if i == 0:
            ax1.plot(synd_ts, color="#607D8B", label='Simulated data', linewidth=1.0, alpha=0.8)
        else:
            ax1.plot(synd_ts, color="#607D8B", linewidth=1.0)
    ax1.set_xlim(xmax=1450, xmin=-10)
    ax1.set_ylim(ymax=90000, ymin=50000)

    ax1.set_yticks([50000, 60000, 70000, 80000, 90000])
    ax1.set_xticks([0, 240, 480, 720, 960, 1200, 1440])

    ax1.plot(
        tst_data, color="#3F51B5", label='Observed data', linewidth=2.3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.grid(True)
    ax1.set_ylabel('Total Power (in MW)', {'fontsize': 14})
    handler.insert_legend(ax1)

plt.show()
