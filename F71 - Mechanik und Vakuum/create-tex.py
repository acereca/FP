import pandas as pd
import numpy as np
import uncertainties.unumpy as un

filelist = {
    "24-d1.dat":"24-t1.tex",
    "24-d2.dat":"24-t2.tex",
    "24-d3.dat":"24-t3.tex",
    "24-d4.dat":"24-t4.tex",
    "25-d1.dat":"25-t1.tex",
    "25-d2.dat":"25-t2.tex",
    "25-d3.dat":"25-t3.tex",
}

pd.options.display.float_format = '{:.2f}'.format

for infile, outfile in filelist.items():
    data = np.genfromtxt(infile, dtype=float, skip_header=0, delimiter=',')[2:]
    names = np.genfromtxt(infile, dtype=str, skip_header=0, delimiter=',', autostrip=True)[0]
    units = np.genfromtxt(infile, dtype=str, skip_header=0, delimiter=',', autostrip=True)[1]

    names = names[::2].tolist()
    units = units[::2].tolist()

    for i in range(len(names)):
        #print(units[i])
        names[i] = str(names[i]) + " / " + str(units[i])


    datasets = un.uarray(data[:, ::2], data[:, 1::2])

    f = open(outfile, 'w')
    f.write(pd.DataFrame(datasets, columns=names).to_latex(escape=False))
