import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as un
import uncertainties as uc

# formatting
plt.style.use('bmh')
plt.figure(figsize=(14,7))

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

plat = np.mean((V1/p1/t1)[3:4])
print("{:.2eL}".format(plat))

plt.plot([1e-5,4e-2], [plat.n]*2)
plt.annotate(
    "${:.2eL}".format(plat) + "\\frac{l}{s}$",
    xy=(2e-2, plat.n),
    xycoords='data',
    xytext=(0, 0),
    textcoords='offset points',
    fontsize=14,
    bbox=dict(boxstyle="round",
    fc="1")
)

plt.errorbar(data[:,0], un.nominal_values(V1/p1/t1), fmt='.', xerr=un.std_devs(p1), yerr=un.std_devs(V1/p1/t1))
plt.errorbar(data2[:,0], un.nominal_values(V2/t2/p2), fmt='.', xerr=un.std_devs(p2), yerr=un.std_devs(V2/p2/t2))

plt.xlabel('p / mbar')
plt.ylabel('S / $\\frac{l}{s}$')
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
    #print(po)
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

    # constant fit
    pu_best = np.where(np.logical_and(pu >= 2e-4, pu <= 2e-3))
    L_fit = np.mean(L[pu_best])
    plt.plot(
        un.nominal_values(pu),
        [L_fit.n]*pu.size,
        label='Fit für $L_{{{0}}}$'.format(firstline)
    )
    print("L_{{{0}}} = {1}".format(firstline,L_fit))
    plt.annotate('${:.2eL}$'.format(L_fit) + " $\\frac{l}{s}$",
                 xy=(1e-2, L_fit.n),
                 xycoords='data',
                 xytext=(0, (5 if L_fit.n > .65 else -10)), textcoords='offset points', fontsize=14,
                 bbox=dict(boxstyle="round", fc="1")
                )


plt.legend()
plt.title('Leiwertbestimmung von Rohr und Blende')
plt.ylabel('L / $\\frac{l}{s}$')
plt.xlabel('p$_{unten}$ / mbar')
plt.savefig('25-f1.png')

LR = uc.ufloat(.659,.032)
LB = uc.ufloat(2.13,.08)
LRB = uc.ufloat(.524,.029)

print("LRB {}".format((LR*LB)/(LR+LB)))
print((LRB -(LR*LB)/(LR+LB)).n/(LRB -(LR*LB)/(LR+LB)).s)

# theo Leitwerte
# lam R
rR = uc.ufloat(12e-3,.1e-3)/2 #m
eta = 17.1e-8 # hPa/s
l = uc.ufloat(1,.05) # m
dp1 = (uc.ufloat(3.1e-1,.1e-1)
    +uc.ufloat(1.1e-2,.1e-2))/2 # mbar
LTR1 = np.pi/8*rR**4*dp1/eta/l
print("LTR1: {} l/s".format(LTR1*1000))
diff = uc.ufloat(.659,.032)-LTR1*1000
print("sig LTR1: {} sigma".format(
        diff.n / diff.s
    )
)
# mol R
R = 8.314 # J/mol/K
M = 28.96e-3 # kg/mol
LTR2 = 8/3*rR**3/l*np.sqrt(np.pi*R*293.15/2/M)
print("LTR2: {} l/s".format(LTR2*1000))
diff2 = uc.ufloat(.659,.032)-LTR2*1000
print("sig LTR2: {} sigma".format(
    diff2.n/diff2.s
))

# mol B
rB = uc.ufloat(4.2e-3,.1e-3)/2 # m
LTB = 362 * rB**2
print("LTB: {} l/s".format(LTB*1000))
diff3 = uc.ufloat(2.13,.08) - LTB*1000
print("sig LTB: {} sigma".format(
    diff3.n/diff3.s
))

LTRB = (LTR1*LTB)*1e3/(LTR1+LTB)
print("LTRB {}".format(LTRB))
print("Abw: {}".format((LRB-LTRB).n/(LRB-LTRB).s))
