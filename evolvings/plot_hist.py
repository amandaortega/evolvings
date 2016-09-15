import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.mlab as mlab
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

for fig, name in enumerate(['10082015', '30112015', '18042016', '27062016']):
    # Initializing testing and the forecast data
    tst_data = np.loadtxt('data/ons/test/%s' % name)
    fcast_data = np.loadtxt('data/ons/fcast/%s' % name)
    error_data = tst_data - fcast_data
    ax1 = plt.subplot(2, 2, fig + 1)

    months = ['August 10, 2015', 'November 30, 2015',
              'April 18, 2016', 'June 27, 2016']
    figure_title = '%s' % months[fig]
    ax1.text(0.5, 1.04, figure_title,
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    #ax1.set_xlim(xmax=1450, xmin=-10)
    #ax1.set_ylim(ymax=2000, ymin=-2000)

    #ax1.set_yticks([-3000, -1500, 0, 1500, 3000])

    mu, sigma = np.average(error_data), np.std(error_data)

    # the histogram of the data
    n, bins, patches = plt.hist(
        error_data, 70, normed=1, alpha=0.00)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    synt_data = calibrate_arma(data=error_data)
    plt.plot(bins, y, linestyle='--', color='purple',
             linewidth=1.5, label='Observed')

    mu, sigma = np.average(synt_data), np.std(synt_data)
    n, bins, patches = plt.hist(
        synt_data, 70, normed=1, facecolor='#000073', alpha=0.00)

    synd_ts = MinMaxScaler((0, max(
        fcast_data + synt_data))).fit_transform(fcast_data + synt_data)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)

    ax1.plot(bins, y, linestyle='--', color='#000073',
             linewidth=1.5, label='Simulated with ARMA(2,2)')

    #plt.setp(ax1.get_xticklabels(), visible=False)
    plt.axis([-3000, 3000, 0, 10])
    ax1.set_ylim(ymax=0.003, ymin=0)
    plt.yticks([0.000, 0.00075, 0.0015, 0.00225, 0.003],
               ['0.00%', '0.75%', '1.50%', '2.25%', '3.0%'])

    ax1.set_ylabel('Relative Error Frequency (%)', {'fontsize': 14})
    handler.insert_legend(ax1)

plt.subplots_adjust(left=0.35, bottom=0.04, right=0.98, top=0.93,
                    wspace=0.24, hspace=0.21)
plt.show()
