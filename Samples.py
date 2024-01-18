import numpy as np
import scipy.constants as const
import warnings as wn
from dataclasses import dataclass

import lmfit as fit
import DataAnalysis.Models as mod
import DataAnalysis.ReadWriteFunctions as tool

file = (
    "\\\\uni.au.dk\\Users\\au684834\\Documents\\"
    + "Python\\Basics\\DataAnalysis\\SampleDB.json"
)
file = "SampleDB.json"
file = "C:\\Users\\mmonti\\Documents\\Python\\DataAnalysis\\SampleDB.json"
e = const.e
m0 = const.m_e
vacImp = const.physical_constants["characteristic impedance of vacuum"]
Z0 = vacImp[0]
e0 = const.epsilon_0
c = const.c
hbar = const.hbar
h = const.h
kB = const.Boltzmann


class Sample:
    """A simple Sample class implementation, useful for quick calculations"""

    def __init__(self, name: str, d: str = 0, tau: str = 0, N: str = 1e17):
        self.para = tool.ReadJSON(file)[name]
        self.N = N * 1e6
        self.d = d
        self.tau = tau

        self.Extraction()
        # self.Calc()

    def __str__(self):
        return (
            "\n"
            + str(self.name)
            + ":\n"
            + "n="
            + str(self.n)
            + ",\nd="
            + str(self.d)
            + "\nn_sub= "
            + str(self.ns)
            + "\nm/m0="
            + str(self.massEffRatio)
            + "\nEg="
            + str(self.Eg)
            + "\n"
        )

    def Extraction(self) -> None:
        """
        Extracts the needed values from the json
        """
        self.massEffRatio = self.para["massR"]
        self.mass = self.massEffRatio * m0
        self.Eg = self.para["Eg"]
        self.eInf = self.para["eInf"]
        self.ns = self.para["ns"]
        self.n = self.para["n"]
        try:
            self.mu_e = self.para["mu_e"]
        except KeyError:
            self.mu_e = 0

    def Calc(self, B: float = 0) -> None:
        """
        Function that computes basic parameters of the sample such as the
        plasma resonance and mobility.
        """
        self.tau_from_mu = e * self.mu_e / self.mass
        self.omC = e * B / self.mass
        self.fC = e * B / (self.mass * 2e12 * np.pi)

        self.cond0 = self.N * self.tau * e**2 / self.mass
        self.omP = np.sqrt(self.N * e**2 / (self.mass * e0))
        self.mobility_from_tau = e * self.tau / self.mass

    def Model(
        self,
        f: np.ndarray = np.linspace(0, 4, 1000),
        model: str = "Drude",
        B: float = 0,
    ) -> dict:
        """
        Does simple modelling of Drude or Cyclotron conductivity.
        """
        om = f * 2e12 * np.pi
        par = fit.Parameters()
        par.add("tau", value=self.tau)
        par.add("N", value=self.N)
        par.add("fC", value=self.fC)

        cond = mod.Switch(model, par, f, mStarRatio=self.massEffRatio, B=B)

        eps = self.eInf + 1j * cond / (om * e0)
        n_complex = np.sqrt(eps)
        T = (1 + self.ns) / (cond * Z0 * self.d + 1 + self.ns)
        output = {"f": f, "cond": cond, "eps": eps, "n": n_complex, "T": T}
        return output


if __name__ == "__main__":
    pass
