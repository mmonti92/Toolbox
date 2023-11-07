import numpy as np
import sympy as sym
import scipy.constants as cnst
import scipy.integrate as spint
import sys
import functools  # stuff for making decorators
import typing  # useful for type annotations

e = cnst.e
kB = cnst.Boltzmann
hbar = cnst.hbar
m0 = cnst.m_e


def Transmission(OD):
    return 10 ** (-OD) * 100  # in %


def OD(T, ref=100.0):
    return -np.log10(T / ref)


def FWHM(s):
    """
    Computes the full width half maximum from the std dev
    """
    return s * 2 * np.sqrt(2 * np.log(2))


def MatDot(A, B, rank, f):
    """
    Matrix multiplication, need some checks on the inputs.
    """
    C = np.zeros((rank, rank, f), dtype=np.complex_)
    for i in range(0, rank):
        for j in range(0, rank):
            for k in range(0, rank):
                C[i, j, :] += A[i, k, :] * B[k, j, :]
    return C


def CtoK(T: float) -> float:
    """
    Handy converter between Celsius to Kelvin. For precise conversion use
    scipy.constants.convert_temperature
    """
    return T + 273.15


def KtoC(T: float) -> float:
    """
    Handy converter between Kelvin to Celsius. For precise conversion use
    scipy.constants.convert_temperature
    """
    return T - 273.15


def isnumber(x: typing.Any) -> bool:
    """
    Checks if x is a number (int, float)
    """
    if type(x) == int or type(x) == float:
        return True
    else:
        return False


def zeropad(
    xData: np.ndarray, yData: np.ndarray, exp: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero pads the pulse

    Parameters
    ----------
    xData: numpy array
        x values either mm or time doesn't matter
    yData: numpy array
        y values
    exp: int
        Expansion factor
        Default: 0

    Returns
    -------
    xOut: extended x values
    yOut: zero-padded y values

    """
    xOut = []
    yOut = []
    if exp > 0:
        step = xData[2] - xData[1]
        xOut = xData
        yOut = yData
        x0 = xData[-1]

        for x in range(exp * len(xData)):
            xOut = np.append(xOut, x0 + x * step)
            yOut = np.append(yOut, 0.0)

    else:
        xOut = xData
        yOut = yData

    return xOut, yOut


def FFT(
    xData: np.ndarray, yData: np.ndarray, xUnit: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the FFT, wrapper around numpy fft

    Parameters
    ----------
    xData: numpy array
        time-axis data (not necessarily in a strictly time unit)
    yData: numpy array
        y data
    xUnit: str
        unit of the x axis,
    """
    # computes the FFT of the data and returns the frequency and the spectrum
    # of the FFT
    # xUnit used to convert the xData in frequency depending on the unit of
    # the x (mm or seconds, not implemented yet)
    yFourier = np.fft.fft(yData)
    fftLen = len(yFourier)
    yFourier = yFourier[0 : int((fftLen / 2 + 1))]
    if xUnit == "OD":
        conv = 0.2998
    elif xUnit == "t":
        conv = 1.0
    else:
        conv = 0.1499

    timeStep = abs(xData[fftLen - 1] - xData[0]) / (fftLen - 1) / conv
    freq = np.array(list(range(int(fftLen / 2 + 1)))) / timeStep / fftLen

    return freq, yFourier


def IFFT(
    xData: np.ndarray, yData: np.ndarray, xUnit: str = "mm"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the inverse FFT, wrapper around numpy fft

    Parameters
    ----------
    xData: numpy array
        time-axis data (not necessarily in a strictly time unit)
    yData: numpy array
        y data
    xUnit: str
        unit of the x axis,
    """
    # computes the FFT of the data and returns the frequency and the spectrum
    # of the FFT
    # xUnit used to convert the xData in frequency depending on the unit of
    # the x (mm or seconds, not implemented yet)
    yIFourier = np.fft.ifft(yData)
    fftLen = len(yIFourier)
    yIFourierOut = yIFourier[0 : int((fftLen / 2 + 1))]
    if xUnit == "OD":
        conv = 0.2998
    elif xUnit == "mm":
        conv = 0.1499
    elif xUnit == "ps":
        conv = 1.0
    elif xUnit == "nm":
        conv = 1.0
    else:
        conv = 0.1499

    timeStep = abs(xData[fftLen - 1] - xData[0]) / (fftLen - 1) / conv
    freq = np.array(list(range(int(fftLen / 2 + 1)))) / timeStep / fftLen

    return freq, yIFourierOut


def SymHessian(f, x, y):
    """
    Calculates the symbolic Hessian
    """
    H = [[[], []], [[], []]]
    H[0][0] = sym.diff(f, x, x)
    H[0][1] = sym.diff(f, x, y)
    H[1][0] = sym.diff(f, y, x)
    H[1][1] = sym.diff(f, y, y)

    return H


def SymGradient(f, x, y):
    """
    Calculates the symbolic gradient
    """
    D = [[], []]
    D[0] = sym.diff(f, x)
    D[1] = sym.diff(f, y)

    return D


def Gradient(x: np.ndarray, step: float = 0) -> np.ndarray:
    """
    Computes the gradient
    """
    xGrad = np.gradient(x)
    if step <= 0:
        step = np.abs(x[-1] - x[-2])
    gradient = xGrad / step

    return gradient


def Hessian(x: np.ndarray, step1: float, step2: float) -> np.ndarray:
    """
    Computes the Hessian
    """
    xGrad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, kgrad in enumerate(xGrad):
        tmp_grad = np.gradient(kgrad)
        for l_k, grad_kl in enumerate(tmp_grad):
            hessian[k, l_k, :, :] = grad_kl / (step1 * step2)
    return hessian


def AntiDerivative(x: np.ndarray, y: np.ndarray):
    """
    Computes the antiderivative (integral)
    """
    out = np.zeros(len(x))
    for i in range(len(x)):
        tmp = 0
        tmp = spint.quad(y, 0, x[i])
        out[i] = tmp[0]
    return x, out


def SymConvolution(f, g, t, lower_limit=-sym.oo, upper_limit=sym.oo):
    """
    Calculates the symbolic convolution between two functions
    """
    tau = sym.Symbol("tau", real=True)
    CI = sym.integrate(
        f.subs(t, tau) * g.subs(t, t - tau), (tau, lower_limit, upper_limit)
    )
    return CI


def MB(E: float, T: float) -> float:
    f = 2 * np.sqrt(E / np.pi) * (1 / (kB * T)) ** 1.5 * np.exp(-E / (kB * T))
    return f


def FD(E: float, mu: float, T: float) -> float:
    return 1 / (1 + np.exp((E - mu) / (kB * T)))


####################################################
