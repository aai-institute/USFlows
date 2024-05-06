import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # plot the same data on both axes
    ax1.scatter([0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.0021, 0.0022],
             [1.154, 1.116, 1.086, 1.142, 1.103,1.096,1.082,1.081,1.096,1.115,16.328,60.363,185.174], c='red')
    ax2.scatter([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013],
             [0.015, 0.020, 0.021, 0.025, 0.023, 0.027, 0.030, 0.029, 0.033, 0.036, 0.035, 0.040, 0.045], c='blue')

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(0.05, 200)  # outliers only
    ax1.set_yticks([1.1, 16, 60, 185])
    ax2.set_ylim(0, 0.05)  # most of the data
    ax2.set_yticks([0.01, 0.02, 0.03, 0.04])

    plt.xlabel('Size of input space',fontsize=18)
    plt.ylabel('Runtime in seconds',fontsize=18)


    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # change the fontsize
    ax1.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='x', labelsize=18)
    plt.yticks(fontsize=18)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.show()


if __name__ == '__main__':
    plt.plot([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013],
             [0.015, 0.020, 0.021, 0.025, 0.023, 0.027, 0.030, 0.029, 0.033, 0.036, 0.035, 0.040, 0.045], 'bo')
    plt.axis((0, 0.015, 0, 0.05))
    plt.xlabel('Size of input space')
    plt.ylabel('Runtime in seconds')
    plt.show()

    plt.plot([0.001, 0.002, 0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013],
             [10.58, 10.43, 10.26, 9.97, 9.59, 9.12, 8.59, 8.00, 7.19, 5.92, 4.23, 2.04, -0.67], 'bo')
    plt.axis((0, 0.015, 0, 11))
    plt.xlabel('Size of input space')
    plt.ylabel('Confidence')
    plt.show()


