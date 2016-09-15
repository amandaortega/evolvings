import numpy as np
import matplotlib.pyplot as plt
from handler import Handler


# Handler class
handler = Handler()

for fig, name in enumerate(['10082015', '30112015', '18042016', '27062016']):
    # Initializing testing and the forecast data
    tst_data = np.loadtxt('data/ons/test/%s' % name)
    fcast_data = np.loadtxt('data/ons/fcast/%s' % name)
    error_data = tst_data - fcast_data
    ax1 = plt.subplot(2, 2, fig + 1)

    months = ['August 10, 2015', 'October 30, 2015',
              'April 18, 2016', 'June 27, 2016']
    figure_title = '%s' % months[fig]
    ax1.text(0.5, 1.04, figure_title,
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)

    ax1.set_xlim(xmax=1450, xmin=-10)
    ax1.set_ylim(ymax=2000, ymin=-2000)

    ax1.set_yticks([-3000, -1500, 0, 1500, 3000])

    ax1.plot(
        error_data, color="red", label='Observed error', linewidth=1.3)

    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.set_ylabel('Forecast Error (in MW)', {'fontsize': 14})
    plt.axhline(0, color='gray', linestyle='--')
    handler.insert_legend(ax1)

plt.subplots_adjust(left=0.35, bottom=0.04, right=0.98, top=0.93,
                    wspace=0.17, hspace=0.15)
plt.show()
