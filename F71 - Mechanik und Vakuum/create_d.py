import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as un

# formatting
plt.style.use('bmh')
plt.figure(figsize=(10,5))

filelist = [
    "24-d1.dat",
    "24-d2.dat"
]

data = np.genfromtxt(filelist[0], dtype=float, delimiter=',')[2:,2:]
data2 = np.genfromtxt(filelist[1], dtype=float, delimiter=',')[2:]
plt.xscale('log')

V1 = un.uarray(data[:,4],data[:,5])
p1 = un.uarray(data[:,0],data[:,1])
t1 = un.uarray(data[:,2],data[:,3])

V2 = un.uarray(data2[:,4],data2[:,5])
p2 = un.uarray(data2[:,0],data2[:,1])
t2 = un.uarray(data2[:,2],data2[:,3])

plt.errorbar(data[:,0], un.nominal_values(V1/p1/t1), fmt='.', xerr=un.std_devs(p1), yerr=un.std_devs(V1/p1/t1))
plt.errorbar(data2[:,0], un.nominal_values(V2/t2/p2), fmt='.', xerr=un.std_devs(p2), yerr=un.std_devs(V2/p2/t2))

plt.savefig('24-f1.png')
plt.clf()
S = ( (V1/p1/t1)[-2] + (V1/p1/t1)[-1] + (V2/p2/t2)[0] )/3
print(S)


filelist = [
    "25-d1.dat",
    "25-d2.dat",
    "25-d3.dat"
]

for infile in filelist:
    data = np.genfromtxt(infile, dtype=float, delimiter=',')[2:]

    plt.xscale('log')

    po = un.uarray(data[:,0], data[:,1])
    pu = un.uarray(data[:,2], data[:,3])

    pd = po -pu


    L = S.n * pu / pd

    #print(infile)
    #print(L)
    print(po)
    with open(infile) as f:
        firstline =f.readline()[2:-1]

    plt.errorbar(
        un.nominal_values(pu),
        un.nominal_values(L),
        fmt='--o',
        label=firstline,
        xerr=un.std_devs(pu),
        yerr=un.std_devs(L)
    )

plt.legend()
plt.savefig('25-f1.png')
