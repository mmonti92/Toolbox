import MyTools as tool
import scipy.constants as cnst
m0 = cnst.m_e

file = '\\\\uni.au.dk\\Users\\au684834\\Documents\\Python\\Basics\\DataAnalysis\\SampleDB.json'

# d = {'InAs': {
#      'n': 3.51, 'ns': 3.5, 'massR': 0.022, 'HHMassR': 0.41,
#      'LHMassR': 0.026, 'eStatic': 15.15, 'eInf': 12.3, 'lattice': 6.0583,
#      'OPE': 0.030, 'Eg': 0.354, 'mu_e': 40000, 'mu_h': 500,
#      'd_e': 1000, 'd_h': 13}}

d = {'InP': {
     'n': 3.5, 'massR': 0.08, 'Eg': 1.344,
     # 'eStatic': 9.72, 'eInf': 6.52, 'lattice': 4.3596,
     # 'OPE': 0.1028, 'Eg': 2.39, 'mu_e': 400, 'mu_h': 50,
     # 'd_e': 20, 'd_h': 8}}
     }}

data = tool.ReadJSON(file)
# # data = d
# print(data)
data.update(d)
# print(data)
tool.WriteJSON(data, file, 'w')