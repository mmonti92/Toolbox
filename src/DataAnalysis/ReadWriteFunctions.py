import numpy as np
import typing as tp
import json
import pickle as pk
import sys
import inspect


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


if __name__ == "__main__":
    pass
