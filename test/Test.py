import THzAnalysis as thz
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

base = '/home/maurizio/Documents/Phd/Data/CsSnI3/April2018/03Apr/'
# dataFiles = ['/home/maurizio/Documents/Phd/Data/CsSnI3/April2018/03Apr/' +
#              'OPTP-RbCsSnI3-SnI2-Q6-500nm-320deg-WF_0.tmp',
#              '/home/maurizio/Documents/Phd/Data/CsSnI3/April2018/03Apr/' +
#              'OPTP-RbCsSnI3-SnI2-Q6-500nm-320deg-WF_1.tmp']
name = 'OPTP-RbCsSnI3-SnI2-Q6-500nm-320deg-WF_'
n = np.arange(0, 9, 1)
dataFiles = []
for i in n:
        j = np.int(np.where(n == i)[0][0])
        dataFiles.append(base + name + str(i) + '.tmp')
mes = thz.THzAnalyser(dataFiles, dataFiles, 'abcd', 'CsSnI3', flip=True,
                      stop=0, start=0, plot=False)

base = '/home/maurizio/Documents/Phd/Data/CsSnI3/March2018/21Mar/'
name = 'OPTP-RbCsSnI3-SnI2-Q6-500nm-320deg-TimeScan-2_1.txt'
dataFiles = [base + name]
mes2 = thz.THzAnalyser(dataFiles, dataFiles, 'abcd', 'CsSnI3', flip=True,
                       stop=0, start=0, plot=False)

f = mes.f
sToT = mes.n
ssr = mes.sigmaUncReal
ssi = mes.sigmaUncImag
# fig = mes.multitimefig
# print(mes.DrudeCoeff)
# r = mes.ratio
f2 = mes2.f
s2 = mes2.sigma
# print(mes2.DrudeCoeff)

# pl0 = plt.errorbar(f, np.real(sToT) / 100, ssr / 100,
#                    ls='', marker='o', label='GaP')
# pl1 = plt.plot(f2, np.real(s2) / 100, ls='', marker='s', label='ZnTe')
# c0 = pl0[0].get_color()
# c1 = pl1[0].get_color()
# plt.errorbar(f, np.imag(sToT) / 100, ssi / 100,
#              ls='', fmt='o', mfc='none', color=c0)
# plt.plot(f2, np.imag(s2) / 100, ls='', marker='s', mfc='none', color=c1)
plt.plot(f, np.imag(sToT))
plt.xlabel('$\\nu$(THz)')
plt.ylabel('$\sigma$(S/cm)')
plt.xlim(0.5, 3)
# plt.ylim(-50, 60)
plt.legend(loc='best')
plt.tight_layout()
# # plt.plot(f, r)
plt.show()
