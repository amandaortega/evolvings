import numpy as np
import matplotlib.pyplot as plt
from repo.epl_krls import ePLKRLSRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.mlab as mlab
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from random import normalvariate


def arrange_data(hh):
    y = hh[24:]
    X = []

    px = np.array([hh[x:x + 24] for x in range(0, len(hh), 24)])[:-1]
    for p in px:
        for i in range(24):
            X.append(p)
    return np.array(X), y

predictors = []
nn = 1

months = ['January', 'April', 'July', 'October']
errors = []
for name in ['012015', '042015', '072015', '102015']:
    data = np.loadtxt('data/pjm/test/%s' % name)

    train = np.loadtxt('data/pjm/train/%s' % name)
    ndata = MinMaxScaler((0, 1)).fit_transform(train)

    X, y = arrange_data(ndata)
    krls = ePLKRLSRegressor()
    for i in range(1, len(X)):
        krls.evolve(X[i], y[i - 1])

    ndata = MinMaxScaler((0, 1)).fit_transform(data)
    X, y = arrange_data(ndata)

    fcast = []
    for i in range(1, len(X)):
        r = krls.evolve(X[i], y[i - 1])
        fcast.append(r)

    denorm_fcast = np.array([int(min(data) + (
        max(data) - min(data)) * k) for k in fcast])
    ax1 = plt.subplot(3, 4, nn)

    figure_title = '%s of 2015' % months[nn - 1]
    plt.text(0.5, 1.08, figure_title,
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    plt.xlim(xmax=725, xmin=0)
    plt.ylim(ymax=7000, ymin=-100)

    plt.plot(
        data[25:], color="blue", label='Observed wind data', linewidth=1.3)
    plt.plot(
        denorm_fcast, color="green", label='ePL-KRLS forecast', linewidth=1.3)

    with open('data/pjm/fcast/%s' % name, 'w+') as w:
        for ir in denorm_fcast:
            w.write(str(ir) + "\n")

    err = data[25:] - denorm_fcast
    ax2 = plt.subplot(3, 4, nn + 4)
    plt.xlim(xmax=725, xmin=0)
    plt.ylim(ymax=2500, ymin=-2500)
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(err, color="red", linewidth=1.0, label='Observed')

    with open('data/pjm/error/%s' % name, 'w+') as w:
        for ir in err:
            w.write(str(ir) + "\n")

    ar, ma = 5, 1

    #step1 = sm.tsa.ARMA(err, (1, 0)).fit()
    #noise = [normalvariate(0, np.std(
    #    step1.resid)) for i in range(len(err))]

    #synt = []
    #y = err[0]
    #for i in range(len(err)):
    #    y = step1.params[1] * y + noise[i]
    #    synt.append(y)

    arma = sm.tsa.ARMA(err, (ar, ma)).fit()
    params = arma.params

    arparams = np.array(params[len(params) - (ar + ma):len(params) - ma])
    maparams = np.array(params[len(params) - ma:])

    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, maparams]

    def gen(nsample):
        return [normalvariate(0, np.std(
            arma.resid)) for i in range(nsample)]

    d_arma = arma_generate_sample(arparams, maparams, len(err), distrvs=gen)

    denorm_arma = MinMaxScaler((min(err), max(err))).fit_transform(d_arma)
    mu, sigma = np.average(err), np.std(err)

    # the histogram of the data
    ax3 = plt.subplot(3, 4, nn + 8)
    n, bins, patches = plt.hist(
        err, 70, normed=1, facecolor='purple', alpha=0.50)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)

    plt.plot(bins, y, linestyle='--', color='purple', linewidth=1.5, label='Observed')

    mu, sigma = np.average(d_arma), np.std(d_arma)
    n, bins, patches = plt.hist(
        d_arma, 70, normed=1, facecolor='#000073', alpha=0.50)

    ax2.plot(d_arma, color='orange', linewidth=0.7, label='Simulated')
    ax1.plot(denorm_fcast + d_arma, color='black', linestyle='--', linewidth=0.7, label='Simulated wind data')

    # Now add the legend with some customizations.
    legend = ax1.legend(loc='upper left', shadow=True)

    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # Now add the legend with some customizations.
    legend = ax2.legend(loc='upper left', shadow=True)

    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)

    plt.plot(bins, y, linestyle='--', color='#000073', linewidth=1.5, label='Simulated with ARMA(5, 1)')

    # Now add the legend with some customizations.
    legend = ax3.legend(loc='upper left', shadow=True)

    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    if nn == 1:
        ax1.set_ylabel('Total Power (in MW)', {'fontsize': 14})
        ax2.set_ylabel('Forecast Error (in MW)', {'fontsize': 14})
        ax3.set_ylabel('Relative Error Frequency (%)', {'fontsize': 14})
    else:
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

    ax1.set_yticks([0, 1500, 3000, 4500, 6000])
    ax2.set_yticks([-2000, -1000, 0, 1000, 2000])
    ax3.set_yticks([0.000, 0.0005, 0.001, 0.0015, 0.002])
    plt.yticks([0.000, 0.00075, 0.0015, 0.00225, 0.003],
               ['0.00%', '0.75%', '1.50%', '2.25%', '3.0%'])
    plt.axis([-1500, 1500, 0, 10])
    plt.ylim(ymax=0.003, ymin=0)
    nn += 1

plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelweight'] = 'bold'
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.95,
                    wspace=0.05, hspace=0.15)
plt.show()
