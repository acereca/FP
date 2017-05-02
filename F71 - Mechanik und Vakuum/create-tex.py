import pandas as pd
import numpy as np
import uncertainties.unumpy as un

def form(x):
    return "${:.2eL}$".format(x)

def form2(x):
    return "${:.2fL}$".format(x)

filelist = {
    "24-d1.dat":"24-t1.tex",
    "24-d2.dat":"24-t2.tex",
    "25-d1.dat":"25-t1.tex",
    "25-d2.dat":"25-t2.tex",
    "25-d3.dat":"25-t3.tex",
}

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

    df = pd.DataFrame(datasets, columns=names)

    print(len(datasets[1]))


    f = open(outfile, 'w')
    f.write(df.to_latex(escape=False, formatters=[form]*len(datasets[1]), index = False))

    print("successfully written data from " + infile + " to " + outfile)
