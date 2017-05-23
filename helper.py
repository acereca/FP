from uncertainties import ufloat
import IPython.display as disp


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
    string = '{name} = '.format(name=name)

    string += '{num:0{width}.{comma}eL}'.format(num=unc, width=width, comma=aftercomma)
    #string += '\pm{num:0{width}.{comma}f})'.format(num=unc.s, width=width, comma=aftercomma)

    string += '\  ' + unit
    disp.display(disp.Math(string))

def plot_prep(title='', xlabel='x', ylabel='y', style='bmh', xscale='linear', yscale='linear', plot_scale=1.5, plot_height=10):
    """
        prepares matplotlib.pyplot with custom formatting
    """

    import matplotlib.pyplot as plt
    plt.figure(figsize=(plot_scale*plot_height,plot_height))
    plt.style.use(style)

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def urepr(name: str, unc: ufloat, unit='', aftercomma=2, addwidth=1, latex=True):
    """
        returning values given as uncertainties.ufloat, for jupyter notebook
    """
    width = aftercomma+addwidth+1
    string = '{name} = '.format(name=name)

    string += '{num:0{width}.{comma}e'
    if latex:
        string += 'L}'
    else:
        string += 'P}'

    string = string.format(num=unc, width=width, comma=aftercomma)
    string += '\  ' + unit

    return string

def prepr(name: str, nom: float, stdd=0, unit='', formatting='f', aftercomma=2, addwidth=1, latex=True):
    """
        pretty printing values given as seperate nominal and stddev values for jupyter notebook
    """
    width = aftercomma+addwidth+1

    string = '{name} = '.format(name=name)

    if stdd != 0:
        string += '('

    string += '{num:0{width}.{comma}f}'.format(num=nom, width=width, comma=aftercomma)

    if stdd != 0:
        if latex:
            string += '\pm{num:0{width}.{comma}{fmt}})'.format(num=stdd, width=width, comma=aftercomma, fmt=formatting)
            string += '\ '
        else:
            string += '±{num:0{width}.{comma}{fmt}})'.format(num=stdd, width=width, comma=aftercomma, fmt = formatting)
            string += ' '

    string += unit
    return string


class OutputTable():
    __list = []

    def add(self, name: str, nom: float, stdd=0, unit='', aftercomma=2, addwidth=1):

        self.__list.append({
            'name': name,
            'nom': nom,
            'stdd': stdd,
            'unit': unit,
            'aftercomma': aftercomma,
            'addwidth': addwidth
        })

    def print(self):
        out = r'\begin{align}' + '\n'

        for entry in self.__list:
            inter = '\t' + entry['name'] + r' &= '
            inter += r'\SI{' + '{:e}'.format(entry['nom']) + r'\pm 2' + '{:e}'.format(entry['stdd']) + r'}'
            inter += r'{' + entry['unit'] + r'}\\' + '\n'

            out += inter

        out += r'\end{align}'

        print(out)
        return out

    def display(self):
        out = self.display()
        disp.display(disp.Math(out))

    def empty(self):
        self.__list.clear()