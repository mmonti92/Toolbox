import DataAnalysis.THzAnalysisImproved as thz
import numpy as np
import matplotlib.pyplot as plt
import DataAnalysis.Samples as sam


# x = np.linspace(-2, 2, 1000)
# xRef = x

# E = np.exp(-(x**2) / 0.005) * np.sin(2 * x)
# ERef = E * 0.5 + 0.01 * np.sin(10 * x) * np.exp(-x / 0.4) * np.heaviside(x, 0)
# # plt.plot(x, E)
# # plt.plot(x, ERef)
# np.savetxt("TestData.txt", np.transpose([x, E]), delimiter="\t")
# np.savetxt("TestDataRef.txt", np.transpose([x, ERef]), delimiter="\t")
sample = sam.Sample("GaAs", 100e-9, 100e-15, 1e17)

analysis = thz.THzAnalysis(sample, "")
analysis.AddFile("TestData.txt", "TestDataRef.txt")
# data = analysis.LoadData("TestData.txt", "TestDataRef.txt")
analysis.CalcQuantities()
print(analysis.GetAverageQuantities())

# plt.plot(data.f, np.abs(data.sigma))

plt.show()
