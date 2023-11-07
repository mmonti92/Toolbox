import numpy as np
import scipy.constants as const
import warnings as wn

import lmfit as fit

e = const.e
m0 = const.m_e
vacImp = const.physical_constants['characteristic impedance of vacuum']
Z0 = vacImp[0]
e0 = const.epsilon_0
c = const.c
hbar = const.hbar
h = const.h
kB = const.Boltzmann
# def Switch(name, par, x, mStarRatio, para=0, para2=0, para3=0):
#     paras = [mStarRatio, para, para2, para3]
#     mapper = {'Drude': Drude,
#               'Cyclotron': Cyclotron,
#               'fC': fC,
#               'DrudeNonP': DrudeNonP,
#               'DoubleDrude': DoubleDrude,
#               'Line': Line,
#               'TransThin': TransmissionDrudeThin,
#               'TransFull': TransmissionDrudeFull,
#               'PhotoTransmission': PhotoTransmission,
#               'QT': QuantumTunneling,
#               'Drude-Smith': Drude_Smith,
#               'Tot': Tot}
#     return mapper[name](par, x, paras)


def Switch(name, par, x, *args, **kwargs):
    mapper = {'Drude': Drude,
              'Cyclotron': Cyclotron,
              'fC': fC,
              'DrudeNonP': DrudeNonP,
              'DoubleDrude': DoubleDrude,
              'Line': Line,
              'TransThin': TransmissionDrudeThin,
              'TransFull': TransmissionDrudeFull,
              'PhotoTransmission': PhotoTransmission,
              'QT': QuantumTunneling,
              'Drude-Smith': Drude_Smith,
              'Tot': Tot,
              'CyclotronTransmission': CyclotronTransmission,
              'DoubleCyclotronTransmission': DoubleCyclotronTransmission, }
    return mapper[name](par, x, *args, **kwargs)


def Line(par, x, *args, **kwargs):
    val = par.valuesdict()
    A = val['A']
    B = val['B']

    line = A * x + B
    return line


def Drude(par, x, paras):
    val = par.valuesdict()
    tau = val['tau']
    N = val['N']
    mStarRatio = paras[0]
    mStar = mStarRatio * const.m_e

    sigma = (N * e**2 * tau / mStar) /\
        (1 - x * 2e12j * np.pi * tau)

    return sigma


def Cyclotron(par, x, B):
    # B = paras[1]
    val = par.valuesdict()
    sigma0 = val['N']
    tau = val['tau']
    omegaC = val['fC'] * 1e12 * 2 * np.pi

    # sigma = (N * e**2 * tau / (m0 * mStarRatio)) *\
    #         (1 - 1j * x * 1e12 * 2 * np.pi * tau) /\
    #         ((1 - 1j * 2 * np.pi * x * tau * 1e12)**2 +
    #          (e * B * tau / (m0 * mStarRatio))**2)
    sigma = sigma0 * (1 - 1j * 2 * np.pi * x * tau * 1e12) /\
        ((1 - 1j * 2 * np.pi * x * tau * 1e12)**2 + omegaC**2 * tau**2)
    return sigma


def CyclotronTransmission(par, x, paras):
    val = par.valuesdict()
    A = val['A']
    gamma = val['gamma'] * 2e12 * np.pi
    omegaC = val['fC'] * 2e12 * np.pi
    omega = x * 2e12 * np.pi

    L = A * 0.5 * gamma / ((omega - omegaC)**2 + (0.5 * gamma)**2)
    T = 1 - L
    return T


def DoubleCyclotronTransmission(par, x, paras):
    val = par.valuesdict()
    A = val['A']
    gamma = val['gamma'] * 2e12 * np.pi
    omegaC = val['fC'] * 2e12 * np.pi
    B = val['B']
    gamma2 = val['gamma2'] * 2e12 * np.pi
    omegaC2 = val['fC2'] * 2e12 * np.pi
    omega = x * 2e12 * np.pi

    L = A * 0.5 * gamma / ((omega - omegaC)**2 + (0.5 * gamma)**2) +\
        B * 0.5 * gamma2 / ((omega - omegaC2)**2 + (0.5 * gamma2)**2)
    T = 1 - L
    return T


def Drude_Smith(par, x, paras):
    val = par.valuesdict()
    tau = val['tau']
    N = val['N']
    c1 = val['c1']
    mr = paras['mr']

    mStar = mr * const.m_e

    sigma = ((N * e**2 * tau / mStar) /
             (1 - x * 2e12j * np.pi * tau)) * (1 + c1 /
                                               (1 -
                                                2e12j * np.pi * x * tau))
    return sigma


def ColeDavidson(par, x, paras):
    val = par.valuesdict()
    N = val['N']
    tau = val['tau']
    b = val['b']
    mr = paras['mr']

    mStar = mr * const.m_e

    sigma = (N * e**2 * tau / mStar) / (1 - 2e12j * np.pi * x * tau)**b
    return sigma


def Lorentz(par, x):
    val = par.valuesdict()
    A = val['A']
    gamma = val['gamma']
    f0 = val['f0']
    om = 2e12 * np.pi * x
    om0 = f0 * np.pi * 2e12

    L = A * 0.5 * gamma / ((om - om0)**2 + (0.5 * gamma)**2)
    return L


def QuantumTunneling(par, x, T):
    val = par.valuesdict()
    tau = val['tau']
    Nr2 = val['N']  # is in fact N*r**2
    T = T[1]

    sigma_t = -Nr2 * e**2 / (6 * kB * T)
    om = x * np.pi * 2e12
    sigma = sigma_t * 1j * om / (np.log(1 - 1j * om * tau))
    return sigma


def Tot(par, x, mStarRatio, T):
    val = par.valuesdict()
    tau = val['tau']
    tau_t = val['tau_t']
    N_tr2 = val['N_t']  # is in fact N*r**2
    f = val['f']
    tau = val['tau']
    N = val['N']

    om = x * np.pi * 2e12
    mStar = m0 * mStarRatio
    sigma_t0 = -N_tr2 * e**2 / (6 * kB * T)
    sigma_t = sigma_t0 * 1j * om / (np.log(1 - 1j * om * tau_t))
    sigma_d0 = N * e**2 * tau / (mStar)
    sigma_d = sigma_d0 / (1 - 1j * om * tau)
    sigma = 1 / (f / sigma_d + (1 - f) / sigma_t)
    return sigma


def DrudeNonP(par, x, paras):
    val = par.valuesdict()
    tau = val['tau']
    N = val['N']
    mStar = paras[0] * const.m_e
    alpha = 1 / paras[1]
    hbar = const.hbar
    Ef = hbar**2 * (3 * np.pi * N)**0.67 / (2 * mStar * e)
    # Ef = 0.25

    sigma = (N * e**2 / mStar) *\
        (tau / (1 - x * 1e12 * 2 * np.pi * tau * 1j)) *\
        ((1 + alpha * Ef)**1.5 /
         (1 + 2 * alpha * Ef))

    # sigma = N * tau / (1 - 2e12j * x * tau)

    # sigma = (e**2 * np.sqrt(mStar) / (3 * np.pi**2 * hbar**3)) *\
    #         (tau / (1 - 2e12j * np.pi * x * tau)) *\
    #         ((N * e)**1.5 * (1 + alpha * N)**1.5 / (1 + 2 * alpha * N))
    return sigma


def DoubleDrude(par, x):
    val = par.valuesdict()
    sigma0 = val['sigma0']
    tau1 = val['tau1']
    tau2 = val['tau2']

    sigma = sigma0 * (tau1 / (1 - 1j * 2 * np.pi * x * tau1 * 1e12)) *\
        (tau2 / (1 - 1j * 2 * np.pi * x * tau2 * 1e12))
    return sigma


def fC(par, P):
    val = par.valuesdict()
    fC0 = val['fC0']
    r = val['Ef']
    A = r**2 * np.pi
    phi = P * 1e-3 * 800e-9 / (5e3 * A * h * c * 1e4)
    N = phi / 1e-5
    Ef = hbar**2 * (3 * np.pi**2 * N * 1e6)**0.667 / (2 * (0.018 * m0) * e)
    fC = fC0 / np.sqrt(1 + 4 * Ef / 0.237)
    return fC


def TransmissionDrudeThin(par, x, mStarRatio, d):
    val = par.valuesdict()
    N = val['N']
    tau = val['tau']
    om = x * 2e12j * np.pi
    ni = 3
    nk = 1
    m = mStarRatio * m0
    t = (ni + nk) * (1 - om * tau) /\
        ((N * e**2 * Z0 * d * tau / m) +
         (ni + nk) * (1 - om * tau))
    return t


def TransmissionDrudeFull(par, x, paras):
    val = par.valuesdict()
    N = val['N']
    tau = val['tau']
    mStarRatio = paras[0]
    d = paras[1]
    om = x * 2e12 * np.pi
    ni = 3.5
    nk = 1
    nr = 1
    m = mStarRatio * m0
    einf = 15.7
    # Eg = 0.17
    # alpha = 1 / Eg
    # Ef = hbar**2 * (3 * np.pi * N)**0.67 / (2 * m * e)
    # Ef = 0.25

    def nj(N, tau, om, m):
        s0 = N * e**2 / m
        nj = (np.sqrt((1j * s0 / e0) *
              (tau / (om - 1j * om**2 * tau)) + einf))
        # nj = np.sqrt(einf +
        #              (1j / (om * e0)) *
        #              (N * e**2 / m) *
        #              (tau / (1 - om * tau * 1j)) *
        #              ((1 + alpha * Ef)**1.5 /
        #               (1 + 2 * alpha * Ef)))
        return nj

    def Phi(om, d, nj):
        phi = om * d * nj / c
        return phi

    nj = nj(N, tau, om, m)
    cosj = np.cos(Phi(om, d, nj))
    sinj = np.sin(Phi(om, d, nj))
    cos1 = np.cos(Phi(om, d, 1))
    sin1 = np.sin(Phi(om, d, 1))
    t = ((ni + nk) * cos1 * nr - 1j * (ni * nk + nr**2) * sin1) *\
        (nj / nr) /\
        (nj * (ni + nk) * cosj - 1j * (ni * nk + nj**2) * sinj)  # *\
# np.exp(-1j * Phi(x, 20e-6, 3.5))
    return np.abs(t)


def PhotoTransmission(par, x, paras):
    val = par.valuesdict()
    N = val['N']
    tau = val['tau']
    mStarRatio, d, tauS, NS = paras[0], paras[1], paras[2], paras[3]
    om = x * 2e12 * np.pi
    ni = 1
    nk = 3.55
    # nr = 1
    m = mStarRatio * m0
    einf = 15.7
    d2 = 6.5e-6
    # Eg = 0.17
    # alpha = 1 / Eg
    # Ef = hbar**2 * (3 * np.pi * N)**0.67 / (2 * m * e)

    def nC(N, tau, om, m):
        s0 = N * e**2 / m
        nj = (np.sqrt((1j * s0 / e0) *
              (tau / (om - 1j * om**2 * tau)) + einf))
        # nj = np.sqrt(einf +
        #              (1j / (om * e0)) *
        #              (N * e**2 / m) *
        #              (tau / (1 - om * tau * 1j)) *
        #              ((1 + alpha * Ef)**1.5 /
        #               (1 + 2 * alpha * Ef)))
        return nj

    def Phi(om, d, nj):
        phi = om * d * nj / c
        return phi

    nj = nC(NS, tauS, om, m)
    cosj = np.cos(Phi(om, d, nj))
    sinj = np.sin(Phi(om, d, nj))
    # nk = nj
    nr = 3.55
    nl = nC(N, tau, om, m)
    cosl = np.cos(Phi(om, d2, nl))
    sinl = np.sin(Phi(om, d2, nl))

    t = ((ni * nj + nj * nr) * cosj - 1j * (ni * nr + nj**2) * sinj) *\
        (nl / nj) /\
        ((ni * nl + nl * nk) * cosl - 1j * (ni * nk + nl**2) * sinl)
    return t
