import numpy as np
import typing as tp

import DataAnalysis.Models as mod  # my models for fitting


def AddError(data: np.ndarray, err: np.ndarray) -> np.ndarray:
    """
    Legacy code
    Sets the data and error in an array for Residual_wrapper
    """
    dataFit = np.append(data, err, axis=0)
    dataFit = np.reshape(dataFit, (2, len(data)))
    return dataFit


def Residual_wrapper(name: str) -> tp.Callable:
    """
    Function that creates a residual to minimize by lmfit,
    choosing different models
    """

    def Residual(
        par: dict,
        x: np.ndarray,
        *args,
        data: np.ndarray = None,
        err: np.ndarray = None,
        **kwargs
    ) -> np.ndarray:
        model = mod.Switch(name, par, x, *args, **kwargs)
        if data is None:
            return model.view(float)
        resid = model - data
        if err is None:
            return resid.view(float)
        resid = np.sqrt(resid**2 / err**2)
        return resid.view(np.float)

    return Residual


if __name__ == "__main__":
    pass
