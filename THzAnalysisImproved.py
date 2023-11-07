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


@dc.dataclass
class FileStructure:
    signalColumn: int = 0
    referenceColumn: int = 0
    skipHeader: int = 0
    delimiter: str = "\t"
    units = "mm"
    conversion = 0.1499


class ComplexData:
    def __init__(
        self,
        Et: np.ndarray,
        t: np.ndarray,
        EtRef: np.ndarray,
        tRef: np.ndarray,
        x: np.ndarray,
        xRef: np.ndarray,
        sample: sam.Sample,
        units: str = "t",
    ):
        self.Et = Et
        self.EtRef = EtRef
        self.t = t
        self.tRef = tRef

        self.x = x
        self.Ref = xRef

        self.units = units
        self.sam = sample

        self.CalcFFT(self)
        self.CalcQuantities(self)

    def CalcFFT(self):
        self.f, self.Ef = mt.IFFT(self.x, self.Et, self.units)
        self.fRef, self.EfRef = mt.IFFT(self.xRef, self.EtRef, self.units)

    def CalcQuantities(self):
        self.trans = self.Ef / self.EfRef
        self.sigma = (
            -(self.sam.ns + self.sam.n2)
            * (self.Ef - self.EfRef)
            / (Z0 * self.sam.d * self.EfRef)
        )
        self.epsilon = self.sam.eInf + 1j * self.sigma / (self.f * 2e12 * np.pi * e0)
        self.loss = np.imag(-1 / self.epsilon)
        self.refractiveIndex = np.sqrt(self.epsilon)


class THzAnalysis:
    """docstring for THzAnalysis"""

    def __init__(self, sample: sam.Sample, fmt: str):
        # super(THzAnalysis, self).__init__()
        self.fileList = []
        self.sample = sample
        self.fmt = fmt

        self.dataList = []
        self.data = 0

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
            d.CalcFFT()
            d.CalcQuantities()

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
                self.fileFormat.units = "OD"
                self.fileFormat.conversion = 0.2998
            case "abcd":
                self.fileFormat.signalColumn = 1
                self.fileFormat.referenceColumn = 2
            case _:
                self.fileFormat.signalColumn = 1
                self.fileFormat.referenceColumn = 12
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

        x, xRef = self.ShiftPeak(x, xRef)
        t = x / self.fileFormat.conversion
        tRef = xRef / self.fileFormat.conversion

        data = ComplexData(
            Et,
            t,
            EtRef,
            tRef,
            x,
            xRef,
            self.sample,
            self.fileFormat.units,
        )
        return data

    def AddFile(self, file: str, refFile: str):
        self.fileList.append([file, refFile])

    def AddData(self, data: ComplexData):
        self.dataList.append(data)

    def CalcQuantities(self):
        for f, fRef in self.fileList:
            data = self.LoadData(f, fRef, self.fmt)
            self.dataList.append(data)

    def AverageData(self, key):
        arr = np.zeros(
            (len(self.dataList, len(vars(self.dataList[0])[key]))), dtype=np.complex
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
            avg[key], std[key] = self.AverageData(self, key)
        return avg, std

    def FitData(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
