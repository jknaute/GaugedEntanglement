from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from matplotlib.transforms import Bbox
from matplotlib import rcParams

def nice_ticks():
    ax = subplot(111)

    ax.get_xaxis().set_tick_params(direction='in', bottom=1, top=1)
    ax.get_yaxis().set_tick_params(direction='in', left=1, right=1)

    # for l in ax.get_xticklines() + ax.get_yticklines():
        # l.set_markersize(10)
        # l.set_markeredgewidth(2.0)
    # for l in ax.yaxis.get_minorticklines() + ax.xaxis.get_minorticklines():
        # l.set_markersize(5)
        # l.set_markeredgewidth(1.5)

    for l in ax.get_xticklines():
        l.set_markersize(8)
        l.set_markeredgewidth(2.0)
    for l in ax.get_yticklines():
        l.set_markersize(8)
        l.set_markeredgewidth(2.0)
    for l in ax.yaxis.get_minorticklines():
        l.set_markersize(4)
        l.set_markeredgewidth(1.5)
    for l in ax.xaxis.get_minorticklines():
        l.set_markersize(4)
        l.set_markeredgewidth(1.5)

    ax.set_position(Bbox([[0.15, 0.15], [0.95, 0.95]]))



linew = 2
rc("font", size = 18) #fontsize of axis labels (numbers)
rc("axes", labelsize = 20, lw = linew) #fontsize of axis labels (symbols)
rc("lines", mew = 2, lw = linew, markeredgewidth = 2)
rc("patch", ec = "k")
rc("xtick.major", pad = 7)
rc("ytick.major", pad = 7)

rcParams["mathtext.fontset"] = "cm"
rcParams["mathtext.rm"] = "serif"
rcParams["figure.figsize"] = [8.0, 6.0]


















