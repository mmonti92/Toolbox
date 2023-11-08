import numpy as np
import dataclasses as dc
import scipy.constants as cnst
import warnings as wn
import typing as tp

import lmfit as fit

import DataAnalysis.MyTools as tool
import DataAnalysis.math_functions as mt
import DataAnalysis.Models as mod
import DataAnalysis.Samples as sam

vacImp = cnst.physical_constants["characteristic impedance" + " of vacuum"]
Z0 = vacImp[0]  # the value of z0
e0 = cnst.epsilon_0
c0 = cnst.c


def ResWrap(modelName: str, *args, **kwargs) -> tp.Callable:
    def Residual(
        par: fit.Parameters, x: np.ndarray, data: np.ndarray = None
    ) -> np.ndarray:
        model = mod.Switch(modelName, par, x, *args, **kwargs)
        if np.any(np.iscomplex(data)):
            model = np.real(model)
        if data is None:
            return model
        dataShape = np.shape(data)

        resid = model - data
        if dataShape[0] < 3:
            resid = model - data[0]
            err = data[1]
            resid = np.sqrt(resid**2 / err**2)
        return resid.view(np.float)

    return Residual


@dc.dataclass
class FileStructure:
    signalColumn: int = 0
    referenceColumn: int = 0
    skipHeader: int = 0
    delimiter: str = "\t"
    conversion = 0.1499


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

        self.UpdateData()

    def CalcFFT(self) -> None:
        self.f, self.Ef = mt.IFFT(self.x, self.Et, "t")
        self.fRef, self.EfRef = mt.IFFT(self.tRef, self.EtRef, "t")

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

        self.epsilon = self.sam.eInf + 1j * self.sigma / (self.f * 2e12 * np.pi * e0)
        self.loss = np.imag(-1 / self.epsilon)
        self.refractiveIndex = np.sqrt(self.epsilon)

    def UpdateData(self) -> None:
        self.CalcFFT()
        self.CalcQuantities()


class THzAnalysis:
    """
    Basic class that implements a thz analysis environment able to read file
    convert them based on given formatting possibilities and calculate basic
    quantities (conductivity, transmission, etc.). Also performs averages and
    basic fitting procedures.

    """

    fileFormatDict: dict = {
        "Oxford": FileStructure(2, 1),
        "Teraview": FileStructure(1, 7, 3, ",", 0.2997),
        "abcd": FileStructure(1, 2),
    }

    def __init__(self, sample: sam.Sample, fmt: str):
        # super(THzAnalysis, self).__init__()
        self.fileList = []
        self.sample = sample
        try:
            self.fileFormat = self.fileFormatDict[fmt]
        except KeyError:
            self.fileFormat = FileStructure(1, 1)
            wn.warn(
                "Warning:: undefined or wrong format, default one chosen: abcd",
                RuntimeWarning,
            )

        self.dataList = []
        self.dataDict = {}
        self.dataErrDict = {}

    def ShiftPeak(self, x, Et, xRef, EtRef) -> tuple[np.ndarray, np.ndarray]:
        M = np.amax(EtRef)
        idx = np.where(EtRef == M)[0][0]
        shift = x[idx]
        x -= shift
        xRef -= shift
        return x, xRef

    def ZeroPad(self, data: ComplexData, exp: int) -> None:
        data.t, data.Et = mt.zeropad(data.t, data.Et, exp)
        data.tRef, data.EtRef = mt.zeropad(data.tRef, data.EtRef, exp)

    def ZeroPadAll(self, exp: int) -> None:
        for d in self.dataList:
            self.ZeroPad(d, exp)
            d.UpdateData()

    def LoadData(self, file: str, refFile: str) -> ComplexData:
        data = tool.Reader(
            file,
            delimiter=self.fileFormat.delimiter,
            skip_header=self.fileFormat.skipHeader,
        )
        refData = tool.Reader(
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

        match self.fmt:
            case "Oxford":
                (EtRef, Et) = (EtRef - 0.5 * Et, EtRef + 0.5 * Et)
                x = x - 24
                xRef = xRef - 24
            case "Warwick":
                Et = EtRef - Et
            case "TeraView":
                pass
            case "abcd":
                pass
            case _:
                wn.warn(
                    "Warning:: undefined or wrong format, "
                    + "default one chosen: abcd",
                    RuntimeWarning,
                )

        x, xRef = self.ShiftPeak(x, Et, xRef, EtRef)
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

    def AddFile(self, file: str, refFile: str) -> None:
        self.fileList.append([file, refFile])

    def CalcQuantities(self) -> None:
        for f, fRef in self.fileList:
            data = self.LoadData(f, fRef)
            self.dataList.append(data)

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
        *args,
        bounds: list = None,
        **kwargs,
    ) -> fit.MinimizerResult:
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
            dataFit = np.append(dataFit, errFit, axis=0)
            dataFit = np.reshape(dataFit, (2, len(errFit)))
        res = ResWrap(model, *args, **kwargs)
        self.guessed = mod.Switch(model, par, self.f, *args, **kwargs)
        out = fit.minimize(
            res,
            par,
            args=(self.f,),
            kws={"data": dataFit},
            nan_policy="omit",
        )
        self.fitted = mod.Switch(model, out.params, self.f, *args, **kwargs)
        return out


if __name__ == "__main__":
    pass
