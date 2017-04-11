from uncertainties import ufloat
import IPython.display as disp
import matplotlib.pyplot as plt

def pprint(name: str, nom: float, stdd=0, unit='', aftercomma=2, addwidth=1):
    """
        pretty printing values given as seperate nominal and stddev values for jupyter notebook
    """
    width = aftercomma+addwidth+1

    string = '{name} = '.format(name=name)

    if stdd != 0:
        string += '('

    string += '{num:0{width}.{comma}f}'.format(num=nom, width=width, comma=aftercomma)

    if stdd != 0:
        string += '\pm{num:0{width}.{comma}f})'.format(num=stdd, width=width, comma=aftercomma)

    string += '\ ' + unit
    disp.display(disp.Math(string))

def uprint(name: str, unc: ufloat, unit='', aftercomma=2, addwidth=1):
    """
        pretty printing values given as uncertainties.ufloat, for jupyter notebook
    """
    width = aftercomma+addwidth+1
    string = '{name} = ('.format(name=name)

    string += '{num:0{width}.{comma}f}'.format(num=unc.n, width=width, comma=aftercomma)
    string += '\pm{num:0{width}.{comma}f})'.format(num=unc.s, width=width, comma=aftercomma)

    string += '\ ' + unit
    disp.display(disp.Math(string))

def plot_prep(title='', xlabel='x', ylabel='y', style='bmh', xscale='linear', yscale='linear', plot_scale=1.5, plot_height=10):
    """
        prepares matplotlib.pyplot with custom formatting
    """
    plt.figure(figsize=(plot_scale*plot_height,plot_height))
    plt.style.use(style)

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
