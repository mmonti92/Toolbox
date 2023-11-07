import numpy as np
import matplotlib.pyplot as plt
import THzAnalysis as thz

import lmfit as fit

import ConductivityModels as mod


base = "/home/maurizio/Documents/Phd/Data/InSb/September2017/19Sep/"


name = ("InSbTMW12007-2K-0p5T-245deg-200ps-CR-EHdetection_0.tmp",
        "InSbTMW12007-2K-0p5T-245deg-200ps-CR-EHdetection_1.tmp",
        "InSbTMW12007-2K-0p5T-245deg-200ps-CR-EHdetection_2.tmp",)
fileList = []
for n in name:
    fileList.append(base + n)
m = 5
M = 24
result = thz.THzAnalyser(fileList=fileList,
                         refList=fileList,
                         fmt='Ox', sampleName='InSb', exp=0,
                         d=200e-9, ns=3.7, plot=False, flip=True,
                         fitFlag=True, model='CyclotronTransmission',
                         para={'B': 0.5}, complexFunction=False,
                         init={'A': 1e10, 'gamma': 0.3, 'fC': 1.0},
                         boundaries=[m, M],
                         fitQty='Transmission')


print(result)
fTheo = np.linspace(0, 4, 1000)
par = fit.Parameters()
par.add('A', value=1e10)
par.add('gamma', value=0.3)
par.add('fC', value=1.0)
# yG = mod.CyclotronTransmission(par, fTheo, paras=[0, 0.5])
y = mod.CyclotronTransmission(result.params, fTheo, paras=[0, 0.5])
plt.close('all')
plt.plot(np.real(result.f), np.abs(result.trans), 'bo', ls='', mfc='none')
plt.plot(np.real(fTheo), np.abs(y), 'b-')
# plt.plot(np.real(fTheo), np.abs(yG), 'b--')
plt.plot(result.f[m], np.abs(result.trans[m]), 'or', ls='')
plt.plot(result.f[M], np.abs(result.trans[M]), 'or', ls='')


# plt.plot(np.real(result.f), np.imag(result.sigma), 'ro', ls='', mfc='none')
# plt.plot(np.real(fTheo), np.imag(y), 'r--')
plt.show()
