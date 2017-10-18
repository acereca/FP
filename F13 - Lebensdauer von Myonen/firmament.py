import uncertainties as unc
import uncertainties.umath as unm
import numpy as np
import VisTools as vt

h = unc.ufloat(17.5, .5)
b = unc.ufloat(32, .5)
l = unc.ufloat(82, .5)

print(4*unm.atan(b*l/(2*h*unm.sqrt(4*h**2+l**2+b**2))))
