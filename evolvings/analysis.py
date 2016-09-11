import numpy as np
import matplotlib.pyplot as plt
from repo.epl_krls import ePLKRLSRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.mlab as mlab


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
    for i in range(len(X)):
        krls.evolve(X[i], y[i])

    ndata = MinMaxScaler((0, 1)).fit_transform(data)
    X, y = arrange_data(ndata)

    fcast = []
    for i in range(len(X)):
        r = krls.evolve(X[i], y[i])
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
    plt.ylim(ymax=6500, ymin=0)

    plt.plot(
        data[24:], color="blue", label='Observed wind data', linewidth=1.3)
    plt.plot(
        denorm_fcast, color="green", label='ePL-KRLS forecast', linewidth=1.3)

    with open('data/pjm/fcast/%s' % name, 'w+') as w:
        for ir in denorm_fcast:
            w.write(str(ir) + "\n")

    # Now add the legend with some customizations.
    legend = ax1.legend(loc='upper left', shadow=True)

    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    err = data[24:] - denorm_fcast
    ax2 = plt.subplot(3, 4, nn + 4)
    plt.xlim(xmax=725, xmin=0)
    plt.ylim(ymax=2000, ymin=-2000)
    plt.axhline(0, color='gray', linestyle='--')
    plt.plot(err, color="red", linewidth=1.0)

    with open('data/pjm/error/%s' % name, 'w+') as w:
        for ir in err:
            w.write(str(ir) + "\n")

    mu, sigma = np.average(err), np.std(err)

    # the histogram of the data
    ax3 = plt.subplot(3, 4, nn + 8)
    n, bins, patches = plt.hist(
        err, 70, normed=1, facecolor='purple', alpha=0.50)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)

    plt.plot(bins, y, 'r--', linewidth=1.5, label='Prob. density function')

    # Now add the legend with some customizations.
    legend = ax3.legend(loc='upper right', shadow=True)

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
    plt.yticks([0.000, 0.0005, 0.001, 0.0015, 0.002],
               ['0.0%', '0.5%', '1.0%', '1.5%', '2.0%'])
    plt.axis([-1500, 1500, 0, 10])
    plt.ylim(ymax=0.002, ymin=0)
    nn += 1

plt.style.use('seaborn-white')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelweight'] = 'bold'
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.98, top=0.95,
                    wspace=0.05, hspace=0.15)
plt.show()
