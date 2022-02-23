import matplotlib.pyplot as plt

def init_plot_style ():
    '''
    Initialize plotting style
    '''

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True


def plot_amp (sim_result):
    '''
    Plot absorption prob in left y-axis and fringing amplitude in right y-axis
    '''
    fig, ax1 = plt.subplots(nrows=1,sharex=True)
    ax2 = ax1.twinx()
    ax1.plot(sim_result,'r-')
    ax1.set_ylabel('Absorption in Silicon')
    ax2.plot(sim_result/np.mean(sim_result)-1,'b-')
    ax2.set_ylabel('Fringing Amplitude')
    ax2.axhline(0,ls = '--',color = 'black')

    return(ax1)


def plot_two(data1,data2, xlabel,ylabel,ls,color,
    figsize = (8,5),label_font = 12,legend_font = 12,ax_c = False):
    '''
    Parameters
    -----------
    data1,data2: 1-d data with same reshape
    xlabel: string
    yabel: list of string
    ls: line style, list of string
    color: plotting color, list of string. Available colors can be found from
           https://matplotlib.org/stable/gallery/color/named_colors.html
    ax_c: The color for the two y axis, list of string

    '''

    fig, ax1 = plt.subplots(nrows=1,sharex=True,figsize = figsize)
    ax2 = ax1.twinx()

    ax1.plot(data1,ls = ls[0], color = color[0], label = label[0])
    ax2.plot(data2, ls = ls[1], color = color[1],label = label[1])

    ax1.legend(frameon = False,fontsize = legend_font, loc = 'best')
    ax2.legend(frameon = False,fontsize= legend_font, loc = 'best')

    ax1.set_ylabel(ylabel[0],fontsize = 12)
    ax2.set_ylabel(ylabel[1],fontsize = 12)
    ax1.set_xlabel(xlabel,fontsize = 12)

    if ax_c != False:
        ax1.spines['left'].set_color(ax_c[0])
        ax2.spines['right'].set_color(ax_c[1])
        ax2.yaxis.label.set_color(ax_c[0])
        ax1.yaxis.label.set_color(ax_c[1])
        ax1.tick_params(axis='y', colors= ax_c[0])
        ax2.tick_params(axis='y',colors = ax_c[0])

    return(fig)

def plot_line (wavelen, intenisty , color):
    for i in range(len(wavelen)):
        plt.plot([wavelen[i],wavelen[i]],[0,intenisty[i]],color = color)
    plt.ylim(0,)
