import sys
from matplotlib import rc
sys.path.append('/home/mmonti/Documents/Python/Programmini/PhD/DataAnalysis/src/')
sys.path.append('/home/mmonti/Documents/Python/Programmini/PhD/DataAnalysis/')
sys.path.append("/home/mmonti/Documents/Python/Programmini/PhD/")

sys.path.append('/home/maurizio/Documents/Python/Programmini/PhD/DataAnalysis/src/')
sys.path.append('/home/maurizio/Documents/Python/Programmini/PhD/DataAnalysis/')
sys.path.append("/home/maurizio/Documents/Python/Programmini/PhD/")

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# defines the font of matplotlib: nedded for using latex
rc('text', usetex=True)  # Allows the use of latex
# rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('text.latex', preamble=r'\usepackage{sfmath}')
# rc('text.latex', preamble=r'\usepackage{bm}')


def Flag(caller=''):
    print('Paths found at ' + caller)
    return 0
