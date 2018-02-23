import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./521.csv')
print(data.columns)
plt.errorbar(
    data['V_i'],
    data['V_o'],
    fmt='.',
    yerr=.02,
    xerr=.02
)

plt.ylabel("V$_{out}$ / kV")
plt.xlabel("V$_{in}$ / V")
plt.show()

