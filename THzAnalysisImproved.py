import numpy as np
import dataclasses as dc
import scipy.constants as cnst
import warnings as wn

import lmfit as fit

import DataAnalysis.MyTools as tool
import DataAnalysis.math_functions as mt
import DataAnalysis.Models as mod
import DataAnalysis.Samples as sam

vacImp = cnst.physical_constants["characteristic impedance" + " of vacuum"]
Z0 = vacImp[0]  # the value of z0
e0 = cnst.epsilon_0
c0 = cnst.c


def ResWrap(modelName: str, *args, **kwargs):
    def Residual(par: fit.Parameters, x: np.ndarray, data: np.ndarray = None):
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

    def CalcFFT(self):
        self.f, self.Ef = mt.IFFT(self.x, self.Et, "t")
        self.fRef, self.EfRef = mt.IFFT(self.tRef, self.EtRef, "t")

    def CalcQuantities(self):
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

    def UpdateData(self):
        self.CalcFFT()
        self.CalcQuantities()


class THzAnalysis:
    """docstring for THzAnalysis"""

    def __init__(self, sample: sam.Sample, fmt: str):
        # super(THzAnalysis, self).__init__()
        self.fileList = []
        self.sample = sample
        self.fmt = fmt

        self.dataList = []
        self.dataDict = {}
        self.dataErrDict = {}

    def ShiftPeak(self, x, Et, xRef, EtRef):
        M = np.amax(EtRef)
        idx = np.where(EtRef == M)[0][0]
        shift = x[idx]
        x -= shift
        xRef -= shift
        return x, xRef

    def ZeroPad(self, data, exp):
        data.t, data.Et = mt.zeropad(data.t, data.Et, exp)
        data.tRef, data.EtRef = mt.zeropad(data.tRef, data.EtRef, exp)

    def ZeroPadAll(self, exp):
        for d in self.dataList:
            self.ZeroPad(d, exp)
            d.UpdateData()

    def LoadData(self, file: str, refFile: str):
        fmt = self.fmt
        self.fileFormat = FileStructure()
        match fmt:
            case "Ox":
                self.fileFormat.signalColumn = 2
                self.fileFormat.referenceColumn = 1

            case "TW":
                self.fileFormat.signalColumn = 1
                self.fileFormat.referenceColumn = 7
                self.fileFormat.skipHeader = 3
                self.fileFormat.delimiter = ","
                self.fileFormat.conversion = 0.2998
            case "abcd":
                self.fileFormat.signalColumn = 1
                self.fileFormat.referenceColumn = 2
            case _:
                self.fileFormat.signalColumn = 1
                self.fileFormat.referenceColumn = 1
                wn.warn(
                    "Warning:: undefined or wrong format, "
                    + "default one chosen: abcd",
                    RuntimeWarning,
                )
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

        match fmt:
            case "Ox":
                (EtRef, Et) = (EtRef - 0.5 * Et, EtRef + 0.5 * Et)
                x = x - 24
                xRef = xRef - 24
            case "Wa":
                Et = EtRef - Et
            case "TW":
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

    def AddFile(self, file: str, refFile: str):
        self.fileList.append([file, refFile])

    def CalcQuantities(self):
        for f, fRef in self.fileList:
            data = self.LoadData(f, fRef)
            self.dataList.append(data)

    def AverageData(self, key: str):
        arr = np.zeros(
            (len(self.dataList), len(vars(self.dataList[0])[key])),
            dtype=complex,
        )
        for i, d in enumerate(self.dataList):
            arr[i] = vars(d)[key]
        avgArr = np.average(arr, axis=0)
        stdArr = np.std(arr, axis=0)
        return avgArr, stdArr

    def GetAverageQuantities(self):
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
        **kwargs,
    ):
        dataFit = self.dataDict[quantity]
        errFit = self.dataErrDict[quantity]

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
