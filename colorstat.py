from __future__ import print_function, division
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def main(filename):
    # load image
    im = Image.open(filename)
    arr = np.rollaxis(np.array(im), 2, 0)
    aspect = arr.shape[1] / arr.shape[2]

    # create plot
    fig = plt.figure(figsize=(10, 6*aspect))

    cmaps = ['Reds_r', 'Greens_r', 'Blues_r']
    colors = [(.9, .1, .1), (.1, .9, .1), (.1, .1, .9)]

    # plot normal figure
    ax = fig.add_axes([0, 0, 3/5, 1])
    ax.imshow(im)
    ax.axis('off')

    for i, a in enumerate(arr):
        # plot each figure band
        ax = fig.add_axes([3/5, (3-(i+1))/3, 1/5, 1/3])
        ax.imshow(a, vmin=0, vmax=255, cmap=cmaps[i])
        ax.axis('off')

        # plot a histogram of each figure band
        # ax.hist was not used, because histogram data was needed
        ax = fig.add_axes([4/5, (3-(i+1))/3, 1/5, 1/3])
        x = np.arange(0, 256)
        hist, _ = np.histogram(a, bins=np.arange(-1, 256)+.5)
        ax.plot(x, hist, 'k-')
        ax.fill_between(x, hist, 0, facecolor=colors[i], lw=0)
        ax.set_xlim(-10, 266)
        ax.set_ylim(0, hist.max()*1.1)
        ax.axis('off')

    # show figure
    plt.show()


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])