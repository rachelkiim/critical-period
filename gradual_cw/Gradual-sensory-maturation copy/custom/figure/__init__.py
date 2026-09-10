import matplotlib.pyplot as plt
import numpy as np

def plot_error(data, color, label=None, xrange=None):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    upper = mean + std
    lower = mean - std

    if xrange is None:
        xrange = np.arange(len(mean))

    if label is None:
        plt.plot(xrange, mean, color=color)
    else:
        plt.plot(xrange, mean, color=color, label=label)

    plt.fill_between(xrange, lower, upper, color=color, alpha=0.2, edgecolor=None)


def bar_error(xticks, data, color):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.bar(xticks, mean, yerr=std, color=color, width=0.5)


mm = 1/2.54/10

color = {
    'darkgray': '#848484',
    'lightgray': '#F7F7F7',
    'orange': '#DF6347',
    'sky': '#0989FF',
    'blue': '#004C98',
    'sand': '#B29166',
    'green': '#495F39',
    'pink': '#B26666',
    'brown': '#634848',
    'purple': '#514863',
}

plt.rcParams['font.family'] = 'Inter'
plt.rcParams['font.style'] = 'normal'

plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['axes.formatter.use_mathtext']=True

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['figure.facecolor'] = (0, 0, 0, 0)
plt.rcParams['axes.facecolor'] = (0, 0, 0, 0)
plt.rcParams['savefig.facecolor'] = (0, 0, 0, 0)

plt.rcParams['figure.figsize'] = (40*mm, 30*mm)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['figure.dpi'] = 180
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.color'] = 'black'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = True

plt.rcParams["boxplot.boxprops.linewidth"] = 0
plt.rcParams["boxplot.whiskerprops.linewidth"] = 0.5
plt.rcParams["boxplot.whiskerprops.linestyle"] = "-"
plt.rcParams["boxplot.whiskerprops.color"] = "black"
plt.rcParams["boxplot.medianprops.linewidth"] = 0.5
plt.rcParams["boxplot.medianprops.color"] = "white"
plt.rcParams["boxplot.capprops.linewidth"] = 0.5
plt.rcParams["boxplot.flierprops.marker"] = "o"
plt.rcParams["boxplot.flierprops.markerfacecolor"] = color["darkgray"]
plt.rcParams["boxplot.flierprops.markersize"] = 3
plt.rcParams["boxplot.flierprops.linestyle"] = "none"
plt.rcParams["boxplot.flierprops.markeredgecolor"] = "none"
plt.rcParams["boxplot.patchartist"] = True
plt.rcParams["boxplot.showfliers"] = False