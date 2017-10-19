import uncertainties as unc
import uncertainties.umath as unm
import numpy as np
import VisTools.printing as vtp

h = unc.ufloat(17.5, .5)
b = unc.ufloat(32, .5)
l = unc.ufloat(82, .5)

firm = 4*unm.atan(b*l/(2*h*unm.sqrt(4*h**2+l**2+b**2)))
vtp.unc_pp("Raumwinkel Omega", firm, unit='sr', formatting='f', aftercomma=3)


area = b*l
vtp.unc_pp("Oberfläche Szintillator", area, unit='cm^2', formatting='f')
vtp.unc_pp("Einfallrate der Myonen",70*area/1e4*firm, 'particle/s', formatting='f', aftercomma=1)
