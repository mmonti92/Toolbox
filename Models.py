import numpy as np
import scipy.constants as const
import scipy.special as sp

import lmfit as fit

e = const.e
m0 = const.m_e
vacImp = const.physical_constants["characteristic impedance of vacuum"]
Z0 = vacImp[0]
e0 = const.epsilon_0
c = const.c
hbar = const.hbar
h = const.h
kB = const.Boltzmann


def Switch(name, par, x, *args, **kwargs):
    mapper = {
        "Line": Line,
        "Poly": Poly,
        "Rise": Rise,
        "Logistic": Logistic,
        "BandPass": BandPass,
        "Decay": Decay,
        "DecayRise": DecayRise,
        "Drop": Drop,
        "DropRise": DropRise,
        "TwoDropRise": TwoDropRise,
        "DropOsc": DropOsc,
        "TwoDropOsc": TwoDropOsc,
        "Dropsh": Dropsh,
        "Lorentz": Lorentz,
        "MultiLorentz": MultiLorentz,
        "DoubleLorentz": DoubleLorentz,
        "Gauss": Gauss,
        "Drude": Drude,
        "Transmission": Trans,
        "Cyclotron": Cyclotron,
        "fC": fC,
        "DrudeNonP": DrudeNonP,
        "DoubleDrude": DoubleDrude,
        "TransThin": TransmissionDrudeThin,
        "TransFull": TransmissionDrudeFull,
        "PhotoTransmission": PhotoTransmission,
        "QT": QuantumTunneling,
        "Drude-Smith": Drude_Smith,
        "Tot": Tot,
        "CyclotronTransmission": CyclotronTransmission,
        "DoubleCyclotronTransmission": DoubleCyclotronTransmission,
    }
    return mapper[name](par, x, *args, **kwargs)


def Line(par, x, *args, **kwargs):
    val = par.valuesdict()
    A = val["A"]
    B = val["B"]

    line = A * x + B
    return line


def Poly(par, x, *args, **kwargs):
    val = par.valuesdict()
    p = 0
    for i, v in enumerate(val):
        p += val[v] * x**i

    return p


def Decay(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    A = val["A"]
    x0 = val["x0"]
    return A * np.exp(-(x - x0) / tau) * np.heaviside(x - x0, 0)


def DecayRise(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    A = val["A"]
    tau2 = val["tau2"]
    B = val["B"]
    x0 = val["x0"]
    return (A * np.exp(-(x - x0) / tau) + B * np.exp(-(x - x0) / tau2)) * np.heaviside(
        x - x0, 0
    )


def Drop(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    A = val["A"]
    x0 = val["t0"]
    tr = val["tr"]
    C = val["C"]
    t = x - x0
    return -A * np.exp(-t / tau) * 0.5 * (1 + sp.erf(t / tr)) + C


def DropRise(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    tauB = val["tauB"]
    # tauD = val["tauD"]
    A = val["A"]
    B = val["B"]
    x0 = val["t0"]
    tr = val["tr"]
    C = val["C"]
    D = val["D"]
    t = x - x0
    return (
        D * (-A * np.exp(-t / tau) + B * np.exp(-t / tauB)) * 0.5 * (1 + sp.erf(t / tr))
    ) + C
    # return -C + np.exp(-t / tauB)


def TwoDropRise(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    tauB = val["tauB"]
    tauE = val["tauE"]
    A = val["A"]
    B = val["B"]
    x0 = val["t0"]
    tr = val["tr"]
    C = val["C"]
    D = val["D"]
    E = val["E"]
    t = x - x0
    return (
        D
        * (-A * np.exp(-t / tau) - E * np.exp(-t / tauE) + B * np.exp(-t / tauB))
        * 0.5
        * (1 + sp.erf(t / tr))
    ) + C


def DropOsc(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    tauB = val["tauB"]
    x0 = val["t0"]
    tr = val["tr"]
    A = val["A"]
    B = val["B"]
    C = val["C"]

    D = val["D"]
    taup = val["taup"]
    w = val["w"]
    p = val["p"]

    E = val["E"]
    w2 = val["w2"]
    taup2 = val["taup2"]
    p2 = val["p2"]

    # F = val["F"]
    # tauF = val["tauF"]
    # w3 = val["w3"]
    # taup3 = val["taup3"]
    # p3 = val["p3"]

    t = x - x0
    return (
        A
        * (
            -np.exp(-t / tau)
            # - F * np.exp(-t / tauF)
            + B * np.exp(-t / tauB)
            + D * np.cos(w * 2 * np.pi * t / 1e3 + p) * np.exp(-t / taup)
            + E * np.cos(w2 * 2 * np.pi * t / 1e3 + p2) * np.exp(-t / taup2)
            # + F * np.cos(w3 * 2 * np.pi * t / 1e3 + p3) * np.exp(-t / taup3)
        )
        * 0.5
        * (1 + sp.erf(t / tr))
    ) + C


def TwoDropOsc(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    tauB = val["tauB"]
    x0 = val["t0"]
    tr = val["tr"]
    A = val["A"]
    B = val["B"]
    C = val["C"]

    D = val["D"]
    taup = val["taup"]
    w = val["w"]
    p = val["p"]

    E = val["E"]
    w2 = val["w2"]
    taup2 = val["taup2"]
    p2 = val["p2"]

    F = val["F"]
    tauF = val["tauF"]
    # w3 = val["w3"]
    # taup3 = val["taup3"]
    # p3 = val["p3"]

    t = x - x0
    return (
        A
        * (
            -np.exp(-t / tau)
            - F * np.exp(-t / tauF)
            + B * np.exp(-t / tauB)
            + D * np.cos(w * 2 * np.pi * t / 1e3 + p) * np.exp(-t / taup)
            + E * np.cos(w2 * 2 * np.pi * t / 1e3 + p2) * np.exp(-t / taup2)
            # + F * np.cos(w3 * 2 * np.pi * t / 1e3 + p3) * np.exp(-t / taup3)
        )
        * 0.5
        * (1 + sp.erf(t / tr))
    ) + C


def Dropsh(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    tauB = val["tauB"]
    x0 = val["t0"]
    tr = val["tr"]
    A = val["A"]
    B = val["B"]
    C = val["C"]

    D = val["D"]
    taup = val["taup"]
    w = val["w"]
    p = val["p"]

    E = val["E"]
    F = val["F"]
    # z = val["zero"]

    t = x - x0
    # Delta = np.exp(-t / tauB) - E * (np.exp(-t / tau) + F)
    Delta = -E * (1 - np.exp(-t / tauB)) * (np.exp(-t / tau) + F)
    beta = np.sin(w * 2 * np.pi * t / 1e3 + p) * np.exp(-t / taup)
    # f = (
    #     A
    #     * (
    #         -np.exp(-t / tau)
    #         # + B * np.exp(-t / tauB)
    #         + (E + D * np.cos(w * 2 * np.pi * t / 1e3 + p) * np.exp(-t / taup)) ** 2
    #         # + E * np.cos(w2 * 2 * np.pi * t / 1e3 + p2) * np.exp(-t / taup2)
    #         # + F * np.cos(w3 * 2 * np.pi * t / 1e3 + p3) * np.exp(-t / taup3)
    #     )
    #     * 0.5
    #     * (1 + sp.erf(t / tr))
    # ) + C

    f = (
        ((A * (-2 * D * Delta + Delta**2) + B * beta**2 + C * (D - Delta) * beta))
        * 0.5
        * (1 + sp.erf(t / tr))
    )
    return f


def Rise(par, t, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    A = val["A"]
    t0 = val["t0"]
    r = val["r"]
    C = val["C"]
    return A * (1 - r * np.exp(-(t - t0) / tau)) * np.heaviside(t - t0, 0) + C


def Logistic(par, x, *args, **kwargs):
    val = par.valuesdict()
    L = val["L"]
    k = val["k"]
    x0 = val["x0"]
    t = x - x0
    return L / (1 + np.exp(-k * t))


def BandPass(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    l0 = val["l0"]
    B = val["B"]
    S = 0
    for i in range(len(val) // 3 - 1):
        par = fit.Parameters()
        par.add("L", value=val["L" + str(i)])
        par.add("k", value=val["k" + str(i)])
        par.add("x0", value=val["x0" + str(i)])
        S += Logistic(par, x)

    e = B * (
        np.exp(-(x - l0) / tau) * np.heaviside(x - l0, 0) + 1 * np.heaviside(l0 - x, 1)
    )

    return S * e


def Lorentz(par, x, *args, **kwargs):
    val = par.valuesdict()
    A = val["A"]
    gamma = val["gamma"]
    f0 = val["f0"]
    C = val["C"]
    # om = 2e12 * np.pi * x
    # om0 = f0 * np.pi * 2e12
    om = x
    om0 = f0

    L = A * 0.5 * gamma / ((om - om0) ** 2 + (0.5 * gamma) ** 2) + C
    return L


def MultiLorentz(par, x, *args, **kwargs):
    val = par.valuesdict()
    C = val["C"]
    if (len(val) - 1) % 3 != 0:
        raise ValueError("Wrong number of parameters")
    L = 0
    for i in range(len(val) // 3):
        A = val["A" + str(i)]
        gamma = val["gamma" + str(i)]
        f0 = val["f0" + str(i)]
        om = x
        om0 = f0
        L += A * 0.5 * gamma / ((om - om0) ** 2 + (0.5 * gamma) ** 2)
    return L + C


def DoubleLorentz(par, x, *args, **kwargs):
    val = par.valuesdict()
    C = val["C"]
    A0 = val["A0"]
    gamma0 = val["gamma0"]
    A1 = val["A1"]
    gamma1 = val["gamma1"]
    f0 = val["f0"]
    om = x
    om0 = f0
    L = A0 * 0.5 * gamma0 / ((om - om0) ** 2 + (0.5 * gamma0) ** 2)
    L += A1 * 0.5 * gamma1 / ((om - om0) ** 2 + (0.5 * gamma1) ** 2)
    return L + C


def Gauss(par, x, *args, **kwargs):
    val = par.valuesdict()
    mu = val["mu"]
    s = val["s"]
    A = val["A"]
    C = val["C"]

    return A * np.exp(-((x - mu) ** 2) / s**2) / (np.sqrt(2 * np.pi) * s) + C


def Drude(par, x, mStarRatio=1.0, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    N = val["N"]
    mStar = mStarRatio * const.m_e

    sigma = (N * e**2 * tau / mStar) / (1 - x * 2e12j * np.pi * tau)

    return sigma


def Trans(par, x, *args, **kwargs):
    val = par.valuesdict()
    tau = val["tau"]
    N = val["N"]
    om = np.pi * 2e12 * x

    t = 1 / (1 + (Z0 * d * N * e**2 * tau / (m * (1 + ns) * (1 + 1j * om * tau))))
    return np.abs(t)


def Cyclotron(par, x, B=0.0, *args, **kwargs):
    val = par.valuesdict()
    sigma0 = val["N"]
    tau = val["tau"]
    omegaC = val["fC"] * 1e12 * 2 * np.pi

    # sigma = (N * e**2 * tau / (m0 * mStarRatio)) *\
    #         (1 - 1j * x * 1e12 * 2 * np.pi * tau) /\
    #         ((1 - 1j * 2 * np.pi * x * tau * 1e12)**2 +
    #          (e * B * tau / (m0 * mStarRatio))**2)
    sigma = (
        sigma0
        * (1 - 1j * 2 * np.pi * x * tau * 1e12)
        / ((1 - 1j * 2 * np.pi * x * tau * 1e12) ** 2 + omegaC**2 * tau**2)
    )
    return sigma


def CyclotronTransmission(par, x, *args, **kwargs):
    val = par.valuesdict()
    A = val["A"]
    gamma = val["gamma"] * 2e12 * np.pi
    omegaC = val["fC"] * 2e12 * np.pi
    omega = x * 2e12 * np.pi

    L = A * 0.5 * gamma / ((omega - omegaC) ** 2 + (0.5 * gamma) ** 2)
    T = 1 - L
    return T


def DoubleCyclotronTransmission(par, x, *args, **kwargs):
    val = par.valuesdict()
    A = val["A"]
    gamma = val["gamma"] * 2e12 * np.pi
    omegaC = val["fC"] * 2e12 * np.pi
    B = val["B"]
    gamma2 = val["gamma2"] * 2e12 * np.pi
    omegaC2 = val["fC2"] * 2e12 * np.pi
    omega = x * 2e12 * np.pi

    L = A * 0.5 * gamma / (
        (omega - omegaC) ** 2 + (0.5 * gamma) ** 2
    ) + B * 0.5 * gamma2 / ((omega - omegaC2) ** 2 + (0.5 * gamma2) ** 2)
    T = 1 - L
    return T


###############################################
# More complex models, not all double-checked #
###############################################


def Drude_Smith(par, x, paras):
    val = par.valuesdict()
    tau = val["tau"]
    N = val["N"]
    c1 = val["c1"]
    mr = paras["mr"]

    mStar = mr * const.m_e

    sigma = ((N * e**2 * tau / mStar) / (1 - x * 2e12j * np.pi * tau)) * (
        1 + c1 / (1 - 2e12j * np.pi * x * tau)
    )
    return sigma


def ColeDavidson(par, x, paras):
    val = par.valuesdict()
    N = val["N"]
    tau = val["tau"]
    b = val["b"]
    mr = paras["mr"]

    mStar = mr * const.m_e

    sigma = (N * e**2 * tau / mStar) / (1 - 2e12j * np.pi * x * tau) ** b
    return sigma


def QuantumTunneling(par, x, T):
    val = par.valuesdict()
    tau = val["tau"]
    Nr2 = val["N"]  # is in fact N*r**2
    T = T[1]

    sigma_t = -Nr2 * e**2 / (6 * kB * T)
    om = x * np.pi * 2e12
    sigma = sigma_t * 1j * om / (np.log(1 - 1j * om * tau))
    return sigma


def Tot(par, x, mStarRatio, T):
    val = par.valuesdict()
    tau = val["tau"]
    tau_t = val["tau_t"]
    N_tr2 = val["N_t"]  # is in fact N*r**2
    f = val["f"]
    tau = val["tau"]
    N = val["N"]

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
    tau = val["tau"]
    N = val["N"]
    mStar = paras[0] * const.m_e
    alpha = 1 / paras[1]
    hbar = const.hbar
    Ef = hbar**2 * (3 * np.pi * N) ** 0.67 / (2 * mStar * e)
    # Ef = 0.25

    sigma = (
        (N * e**2 / mStar)
        * (tau / (1 - x * 1e12 * 2 * np.pi * tau * 1j))
        * ((1 + alpha * Ef) ** 1.5 / (1 + 2 * alpha * Ef))
    )

    # sigma = N * tau / (1 - 2e12j * x * tau)

    # sigma = (e**2 * np.sqrt(mStar) / (3 * np.pi**2 * hbar**3)) *\
    #         (tau / (1 - 2e12j * np.pi * x * tau)) *\
    #         ((N * e)**1.5 * (1 + alpha * N)**1.5 / (1 + 2 * alpha * N))
    return sigma


def DoubleDrude(par, x):
    val = par.valuesdict()
    sigma0 = val["sigma0"]
    tau1 = val["tau1"]
    tau2 = val["tau2"]

    sigma = (
        sigma0
        * (tau1 / (1 - 1j * 2 * np.pi * x * tau1 * 1e12))
        * (tau2 / (1 - 1j * 2 * np.pi * x * tau2 * 1e12))
    )
    return sigma


def fC(par, P):
    val = par.valuesdict()
    fC0 = val["fC0"]
    r = val["Ef"]
    A = r**2 * np.pi
    phi = P * 1e-3 * 800e-9 / (5e3 * A * h * c * 1e4)
    N = phi / 1e-5
    Ef = hbar**2 * (3 * np.pi**2 * N * 1e6) ** 0.667 / (2 * (0.018 * m0) * e)
    fC = fC0 / np.sqrt(1 + 4 * Ef / 0.237)
    return fC


def TransmissionDrudeThin(par, x, mStarRatio, d):
    val = par.valuesdict()
    N = val["N"]
    tau = val["tau"]
    om = x * 2e12j * np.pi
    ni = 3
    nk = 1
    m = mStarRatio * m0
    t = (
        (ni + nk)
        * (1 - om * tau)
        / ((N * e**2 * Z0 * d * tau / m) + (ni + nk) * (1 - om * tau))
    )
    return t


def TransmissionDrudeFull(par, x, paras):
    val = par.valuesdict()
    N = val["N"]
    tau = val["tau"]
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
        nj = np.sqrt((1j * s0 / e0) * (tau / (om - 1j * om**2 * tau)) + einf)
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
    t = (
        ((ni + nk) * cos1 * nr - 1j * (ni * nk + nr**2) * sin1)
        * (nj / nr)
        / (nj * (ni + nk) * cosj - 1j * (ni * nk + nj**2) * sinj)
    )  # *\
    # np.exp(-1j * Phi(x, 20e-6, 3.5))
    return np.abs(t)


def PhotoTransmission(par, x, paras):
    val = par.valuesdict()
    N = val["N"]
    tau = val["tau"]
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
        nj = np.sqrt((1j * s0 / e0) * (tau / (om - 1j * om**2 * tau)) + einf)
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

    t = (
        ((ni * nj + nj * nr) * cosj - 1j * (ni * nr + nj**2) * sinj)
        * (nl / nj)
        / ((ni * nl + nl * nk) * cosl - 1j * (ni * nk + nl**2) * sinl)
    )
    return t


if __name__ == "__main__":
    pass
