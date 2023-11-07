import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd

import lmfit as fit


def Conversion(x, par):
        val = par.valuesdict()
        P0 = val['P0']
        a = val['a']
        P = P0 * 10**(a * x)
        return P


def Residual(par, x, data=None):
                model = Conversion(x, par)
                if data is None:
                        return model
                resid = model - data
                dataShape = np.shape(data)
                if dataShape[0] <= 3:
                    resid = model - data[0]
                    err = data[1]
                    resid = np.sqrt(resid**2 / err**2)
                return resid


df = pd.read_excel('/home/maurizio/Downloads/06062018Calibration.xlsx',
                   '780nm')
fmt = ['E', 'deg']
data = df[fmt]
x = data['deg'].values
y = data['E'].values
x = x[0:-10]
y = y[0:-10]
par = fit.Parameters()
par.add('a', value=0.00741)
par.add('P0', value=0.05)
guess = Conversion(x, par)
out = fit.minimize(Residual, par, args=(x,),
                   kws={'data': y},
                   nan_policy='propagate')
fitted = Conversion(x, out.params)

print(Conversion(215, out.params))


plt.plot(x, y, 'sb', ls='')
plt.plot(x, guess, 'b--')
plt.plot(x, fitted, '-b')
print(fit.fit_report(out))
plt.show()
