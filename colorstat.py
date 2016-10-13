from __future__ import print_function, division
from PIL import Image
import numpy as np
from sklearn import cluster
from scipy.spatial.distance import cdist, euclidean
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import style, patches
style.use('ggplot')


def _show(fig):
    # running = QtGui.QApplication.instance() is not None
    # QtGui.QApplication.instance()
    # if not running:
    #     app = QtGui.QApplication([])

    w = FigureCanvas(fig)
    WIDGETS.append(w)
    w.show()

    # if not running:
    #     raise SystemExit(app.exec_())

    return w


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def _load_fig(filename):
    im = Image.open(filename)
    w, h = im.size
    aspect = h/w
    return im, aspect


def _load_array(filename):
    im, aspect = _load_fig(filename)
    arr = np.rollaxis(np.array(im), 2, 0)
    return arr, aspect


def show(filename, size=10.):
    im, aspect = _load_fig(filename)
    fig = Figure(figsize=(size, size*aspect))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im)
    ax.axis('off')
    return _show(fig)


def split(filename, size=10.):
    arr, aspect = _load_array(filename)

    fig = Figure(figsize=(size, size*aspect))

    cmaps = ['Reds_r', 'Greens_r', 'Blues_r', 'gray_r']
    for i, band in enumerate(arr):
        ax = fig.add_axes([(i % 2) * .5, (1 - i // 2) * .5, .5, .5])
        ax.imshow(band, cmap=cmaps[i])
        ax.axis('off')

    return _show(fig)


def hist(filename):
    arr, aspect = _load_array(filename)

    fig = Figure(figsize=(10, 8))

    colors = [(.6, .1, .1), (.1, .6, .1), (.1, .1, .6), (.1, .1, .1)]
    names = ['r', 'g', 'b', 'a']
    for i, band in enumerate(arr):
        ax = fig.add_subplot(2, 2, i+1)  #add_axes([(i % 2) * .5, (1 - i // 2) * .5, .5, .5])
        x = np.arange(256)
        hist, _ = np.histogram(band, bins=np.arange(-1, 256) + .5)
        hist = hist / np.prod(band.shape)
        ax.fill_between(x, hist, 0, facecolor=colors[i], lw=0)
        ax.set_xlim(-10, 266)
        ax.set_ylim(0, hist.max())
        ax.set_title(names[i])

    return _show(fig)


def main_colors(filename, N=5, sample_size=100000, size=4):
    arr, aspect = _load_array(filename)

    arr_flat = arr.reshape(arr.shape[0], -1)[:3, :]
    random_ind = np.random.randint(0, arr_flat.shape[1], size=sample_size)
    X = arr_flat[:, random_ind].T
    kmeans = cluster.KMeans(n_clusters=N, random_state=0).fit(X)
    colors = []
    for i in range(N):
        cluster_colors = X[kmeans.labels_ == i, :]
        c = geometric_median(cluster_colors)
        colors.append(c)

    fig = Figure(figsize=(size, size/N))
    ax = fig.add_axes([0, 0, 1, 1])
    for i, c in enumerate(colors):
        ax.add_patch(patches.Rectangle((i*(1/N), 0), 1/N, 1, facecolor=c/255, edgecolor="none"))
    ax.axis('off')

    return _show(fig)


def show_all(filename, N=5):
    show(filename)
    hist(filename)
    split(filename)
    main_colors(filename, N=N)


if __name__ == '__main__':
    import argparse
    import inspect
    import sys

    def add_func_parser(subps, fn):
        p = subps.add_parser(fn.__name__.replace('_', '-'))
        spec = inspect.getargspec(fn)
        pos_count = len(spec.args) - len(spec.defaults or ())
        args = spec.args[:pos_count]
        kwargs = zip(spec.args[pos_count:], spec.defaults or ())

        for a in args:
            p.add_argument(a)
        for kw, default in kwargs:
            t = type(default) if default is not None else None
            name = '--'+kw.replace('_', '-')
            p.add_argument(name, default=default, type=t)
        p.set_defaults(func=fn)

    parser = argparse.ArgumentParser()
    subps = parser.add_subparsers()
    for fn in [show, show_all, hist, split, main_colors]:
        add_func_parser(subps, fn)


    app = QtGui.QApplication([])
    WIDGETS = []

    argv = sys.argv[1:]
    args = vars(parser.parse_args(argv))
    w = args.pop('func')(**args)

    raise SystemExit(app.exec_())
