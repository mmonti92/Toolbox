import numpy as np
import dataclasses as dc
import scipy.constants as cnst
import warnings as wn
import typing as tp

import lmfit as fit

import DataAnalysis.ReadWriteFunctions as rw
import DataAnalysis.FittingFunctions as ff
import DataAnalysis.math_functions as mt
import DataAnalysis.Models as mod
import DataAnalysis.Samples as sam

vacImp = cnst.physical_constants["characteristic impedance of vacuum"]
Z0 = vacImp[0]  # the value of z0
e0 = cnst.epsilon_0
c0 = cnst.c


@dc.dataclass(slots=True, frozen=True)
class FileStructure:
    name: str = ""
    signalColumn: int = 0
    referenceColumn: int = 0
    skipHeader: int = 0
    delimiter: str = "\t"
    conversion: float = 0.1499


class ComplexData:
    """
    A class that implements complex electric field data and calculates the
    equivalent quantities (conductivity, transmission etc)
    """

    def __init__(
        self,
        Et: np.ndarray,
        t: np.ndarray,
        EtRef: np.ndarray,
        tRef: np.ndarray,
        sample: sam.Sample,
    ):
        self.Et = Et
        self.EtRef = EtRef
        self.t = t
        self.tRef = tRef

        self.sam = sample

        self.CalcFFT()
        self.CalcQuantities()

    def CalcFFT(self) -> None:
        self.f, self.Ef = mt.FFT(self.t, self.Et, "t")
        self.fRef, self.EfRef = mt.FFT(self.tRef, self.EtRef, "t")

    def CalcQuantities(self) -> None:
        self.trans = self.Ef / self.EfRef
        try:
            n2 = self.sam.n2
        except AttributeError:
            n2 = 1
        self.sigma = (
            -(self.sam.ns + n2)
            * (self.Ef - self.EfRef)
            / (Z0 * self.sam.d * self.EfRef)
        )

        self.epsilon = self.sam.eInf + 1j * self.sigma / (
            self.f * 2e12 * np.pi * e0
        )
        self.loss = np.imag(-1 / self.epsilon)
        self.refractiveIndex = np.sqrt(self.epsilon)


class THzAnalysis:
    """
    Basic class that implements a thz analysis environment able to read file
    convert them based on given formatting possibilities and calculate basic
    quantities (conductivity, transmission, etc.). Also performs averages and
    basic fitting procedures.

    """

    fileFormatDict: dict = {
        "Oxford": FileStructure("Oxford", 2, 1),
        "Teraview": FileStructure("Teraview", 1, 7, 3, ",", 0.2997),
        "abcd": FileStructure("abcd", 1, 2),
        "IMMM": FileStructure("IMMM", 1, 1, conversion=1),
    }

    def __init__(
        self, sample: sam.Sample = sam.Sample("Air"), fmt: str = "IMMM"
    ):
        self.fileList = []
        self.sample = sample
        try:
            self.fileFormat = self.fileFormatDict[fmt]
        except KeyError:
            self.fileFormat = FileStructure(1, 1)
            wn.warn(
                "Warning:: undefined or wrong format"
                + "default one chosen: IMMM",
                RuntimeWarning,
            )

        self.dataList = []
        self.dataDict = {}
        self.dataErrDict = {}

    def AddFile(self, file: str, refFile: str) -> None:
        self.fileList.append([file, refFile])

    def LoadData(
        self, file: str, refFile: str, shift: float = None
    ) -> ComplexData:
        data = rw.Reader(
            file,
            delimiter=self.fileFormat.delimiter,
            skip_header=self.fileFormat.skipHeader,
        )
        refData = rw.Reader(
            refFile,
            delimiter=self.fileFormat.delimiter,
            skip_header=self.fileFormat.skipHeader,
        )

        x, Et, xRef, EtRef = (
            data[0],
            data[self.fileFormat.signalColumn],
            refData[0],
            refData[self.fileFormat.referenceColumn],
        )

        match self.fileFormat.name:
            case "Oxford":
                (EtRef, Et) = (EtRef - 0.5 * Et, EtRef + 0.5 * Et)
                x = x - 24
                xRef = xRef - 24
            case "Warwick":
                Et = EtRef - Et
            case "TeraView" | "abcd" | "IMMM":
                pass
            case _:
                wn.warn(
                    "Warning:: undefined or wrong format, "
                    + "default one chosen: abcd",
                    RuntimeWarning,
                )
        if shift:  # have to add this as applyall
            x, xRef = self.ShiftPeak(x, Et, xRef, EtRef, shift)
        t = x / self.fileFormat.conversion
        tRef = xRef / self.fileFormat.conversion

        data = ComplexData(
            Et,
            t,
            EtRef,
            tRef,
            self.sample,
        )
        return data

    def LoadAll(self, *args: tp.Any, **kwargs) -> None:
        for f, fRef in self.fileList:
            data = self.LoadData(f, fRef, *args, **kwargs)
            self.dataList.append(data)

    def ShiftPeak(
        self,
        x: np.ndarray,
        Et: np.ndarray,
        xRef: np.ndarray,
        EtRef: np.ndarray,
        shift: float = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if shift:
            x -= shift
            xRef -= shift
        else:
            M = np.amax(EtRef)
            idx = np.where(EtRef == M)[0][0]
            shift = x[idx]
            x -= shift
            xRef -= shift
        return x, xRef

    def ZeroPad(self, data: ComplexData, exp: int) -> None:
        data.t, data.Et = mt.zeropad(data.t, data.Et, exp)
        data.tRef, data.EtRef = mt.zeropad(data.tRef, data.EtRef, exp)

    def Chop(self, data: ComplexData, tR: float, tL: float = -np.inf) -> None:
        mask = (tL < data.t) & (data.t < tR)
        data.t, data.Et = data.t[mask], data.Et[mask]
        data.tRef, data.EtRef = data.tRef[mask], data.EtRef[mask]

    def Window(self, data: ComplexData, windowFunction: callable) -> None:
        window = windowFunction(data.t)
        data.Et *= window
        window = windowFunction(data.tRef)
        data.EtRef *= window

    def AppplyAll(self, f: callable, *args: tp.Any, **kwargs: tp.Any) -> None:
        for d in self.dataList:
            f(d, *args, **kwargs)
            d.CalcFFT()
            d.CalcQuantities()

    def AverageData(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        arr = np.zeros(
            (len(self.dataList), len(vars(self.dataList[0])[key])),
            dtype=complex,
        )
        for i, d in enumerate(self.dataList):
            arr[i] = vars(d)[key]
        avgArr = np.average(arr, axis=0)
        stdArr = np.std(arr, axis=0)
        return avgArr, stdArr

    def GetAverageQuantities(self) -> tuple[dict, dict]:
        avg = {}
        std = {}
        for key, val in vars(self.dataList[0]).items():
            if key == "sam":
                pass
            else:
                avg[key], std[key] = self.AverageData(key)
        self.dataDict = avg
        self.dataErrDict = std
        return avg, std

    def FitData(
        self,
        model: str,
        quantity: str,
        par: fit.Parameters,
        *args: tp.Any,
        bounds: list = None,
        **kwargs: tp.Any,
    ) -> fit.Minimizer:
        dataFit = self.dataDict[quantity]
        errFit = self.dataErrDict[quantity]
        if bounds:
            dataFit = dataFit[bounds[0] : bounds[1]]
            errFit = errFit[bounds[0] : bounds[1]]
        else:
            idx = np.where((self.f > 0.2) & (self.f < 3))
            dataFit = dataFit[idx]
            errFit = errFit[idx]

        if np.any(errFit):
            kws = {"data": dataFit, "err": errFit}
        else:
            kws = {"data": dataFit}

        res = ff.ResWrap(model, *args, **kwargs)
        self.guessed = mod.Switch(model, par, self.f, *args, **kwargs)
        out = fit.minimize(
            res,
            par,
            args=(self.f,),
            kws=kws,
            nan_policy="omit",
        )
        self.fitted = mod.Switch(model, out.params, self.f, *args, **kwargs)
        return out


if __name__ == "__main__":
    pass
