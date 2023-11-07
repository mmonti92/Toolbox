############################################
# some useful modules and functions
# Maurizio Monti 2021
###############################################

import numpy as np
import pandas as pd
import sys
import inspect  # contains stuff to inspect module tree
import matplotlib.pyplot as plt
import functools  # stuff for making decorators
import time as tm
import smtplib as sm  # email protocol
import ssl  # ssl protocol
import email.message  # Email formatting tools (e.g. subject, sender, etc.)
import pickle as pk
import json  # interface with json type
import typing as tp  # support for annotations

import DataAnalysis.math_functions as mt  # set of mathematical tools
import DataAnalysis.Models as mod  # my models for fitting


def norm(t: np.ndarray, tMin: float, tMax: float) -> np.ndarray:
    """
    Normalized a value between two numbers, for colormaps
    """
    norm = np.abs(t - tMin) / np.abs(tMax - tMin)
    return norm


def ReadJSON(file: str) -> dict:
    """
    Read a json file
    """
    with open(file) as f:
        data = json.load(f)
    return data


def WriteJSON(data: dict, file: str, mode: str = "w") -> None:
    """
    Writes a dictionary into a json file
    """
    with open(file, mode, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def ReadPickle(file: str) -> tp.Any:
    """
    Read the content of a pickle file
    """
    with open(file, "rb") as pickle_file:
        content = pk.load(pickle_file)
    return content


def SendMail(text: str, subj: str, receiver: str, port: int = 465) -> None:
    """
    Simple script to send an email with my gmail bot
    """
    sender = "MMCodeBot@gmail.com"
    pwd = "rlqjbviptikkwtri"
    msg = email.message.Message()
    msg["Subject"] = subj
    msg["From"] = sender
    msg["To"] = receiver
    msg.add_header("Content-Type", "Text")
    msg.set_payload(text)

    context = ssl.create_default_context()
    with sm.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender, pwd)
        server.sendmail(sender, receiver, msg.as_string())


def Norm(t: float, tMin: float = 0, tMax: float = 1) -> float:
    """
    Returns a given input as normalized between two extrema
    """
    norm = np.abs(t - tMin) / np.abs(tMax - tMin)
    return norm


def print_annotations(func: tp.Callable) -> tp.Callable:
    @functools.wraps(func)
    def Wrapper_annotations(*args, **kwargs):
        print(f"{func.__name__!r} annotations: {tp.get_type_hints(func)}")
        return func(*args, **kwargs)

    return Wrapper_annotations


def timer(func: tp.Callable) -> tp.Callable:
    """
    A simple decorator that compute the function run time
    """

    @functools.wraps(func)
    def Wrapper_timer(*args, **kwargs):
        start = tm.perf_counter()
        value = func(*args, **kwargs)
        end = tm.perf_counter()
        run_time = end - start
        print(f"Finished {func.__name__!r} in {run_time:.4f}s")
        return value

    return Wrapper_timer


def slow_down(t: float) -> tp.Callable:
    """
    A simple decorator that waits a time t before executing the function
    """

    def Slow_down_decorator(func: tp.Callable) -> tp.Callable:
        @functools.wraps(func)
        def Wrapper_slow_down(*args, **kwargs):
            tm.sleep(t)
            return func(*args, **kwargs)

        return Wrapper_slow_down


def AddError(data: np.ndarray, err: np.ndarray) -> np.ndarray:
    """
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
        par: dict, x: np.ndarray, *args, data: np.ndarray = None, **kwargs
    ) -> np.ndarray:
        model = mod.Switch(name, par, x, *args, **kwargs)
        resid = model - data
        if data is None:
            return model
        resid = model - data
        # dataShape = np.shape(data)
        # if dataShape[0] <= 3:  # dumb way to add the option for errors
        #     resid = model - data[0]
        #     err = data[1]
        #     resid = np.sqrt(resid**2 / err**2)
        #     # print(resid)
        return resid

    return Residual


def Reader(
    file: str,
    comments: str = "%",
    delimiter: str = "\t",
    transpose: bool = True,
    *args: tp.Optional,
    **kwargs: tp.Optional,
) -> np.ndarray:
    """
    A wrapper around numpy genfromtxt to read text files

    Parameters
    ----------
    file: string
        full path and name of the file to open
    comment: str
        Symbol in front of comments in the file
        Default: %
    delimiter: str
        file delimiter
        Default: \t
    caller: str
        option to output a string in case of IOError, useful in debugging
    transpose: bool
        If True transpose the data
        Default: True

    Returns
    -------
    data: the column data, unpacked
    """

    caller = inspect.getframeinfo(sys._getframe(1)).filename
    data = 0
    try:
        data = np.genfromtxt(
            file,
            unpack=False,
            comments=comments,
            delimiter=delimiter,
            *args,
            **kwargs,
        )
        if "names" not in kwargs:
            datanew = data[:, ~np.all(np.isnan(data), axis=0)]
            data = 0
        else:
            datanew = data
        if transpose:
            data = np.transpose(datanew)
        else:
            data = datanew
    except IOError:
        raise IOError(f"File {file} not found by {caller}")

    return data


if __name__ == "__main__":
    pass
