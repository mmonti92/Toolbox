import numpy as np
import functools
import DataAnalysis.math_functions as mt
import typing as tp


class NotAVectorError(TypeError):
    """docstring for NotAVectorError"""

    def __init__(self, obj: tp.Any, message: str = "", *args) -> None:
        self.wrong_obj = obj
        self.message = message
        if message == "":
            msg = (
                "Operation not possible between"
                + " object of type Vector"
                + " and object of type "
                + str(type(self.wrong_obj))
            )
        else:
            msg = message
        super().__init__(msg)


class NotANumberError(TypeError):
    def __init__(self, obj: tp.Any, message: str = "", *args) -> None:
        self.wrong_obj = obj
        self.message = message
        if message == "":
            msg = (
                "Operation not possible between"
                + " object of type Vector"
                + " and object of type "
                + str(type(self.wrong_obj))
            )
        else:
            msg = message
        super().__init__(msg)


class AmbiguousOperationError(RuntimeError):
    def __init__(self, message: str = "", *args) -> None:
        self.message = message
        if message == "":
            msg = (
                "Multiplication between two "
                + 'vectors is ambiguous, use "dot"'
                + ' or "cross"'
            )
        else:
            msg = message
        super().__init__(msg)


def test_vec(func):
    @functools.wraps(func)
    def Wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, Vector):
                pass
            elif isinstance(arg, list) or isinstance(arg, np.ndarray):
                try:
                    v = Vector(arg)
                    args = list(args)
                    args[i] = v
                except ValueError:
                    raise NotAVectorError(arg)
            else:
                raise NotAVectorError(arg)
        value = func(*args, **kwargs)
        return value

    return Wrapper


def test_num(func):
    @functools.wraps(func)
    def Wrapper(*args, **kwargs):
        if mt.isnumber(args[1]):
            pass
        elif isinstance(args[1], str):
            try:
                v = float(args[1])
                args = list(args)
                args[1] = v
            except ValueError:
                raise NotANumberError(args[1])

        elif isinstance(args[1], Vector):
            raise AmbiguousOperationError()
        else:
            raise NotANumberError(args[1])
        value = func(*args, **kwargs)
        return value

    return Wrapper


class Vector:
    """This is a class that implements a vector. Basis is a numpy array."""

    __slots__ = "val", "dim", "norm", "theta"

    def __init__(self, val: list = None, dim: int = 0):
        super(Vector, self).__init__()
        if dim == 0 and val is None:
            raise ValueError()
        elif dim != 0 and val is None:
            self.val = np.zeros(dim, dtype="double")
        else:
            self.val = np.array(val, dtype="double")
        if val is not None and len(val) != dim:
            self.dim = len(self.val)
            # should also raise a warning
        else:
            self.dim = dim

        self.norm = self.Norm()
        if self.dim == 2:
            self.theta = self.Theta()
        elif self.dim == 3:
            pass
        else:
            pass

    def __str__(self):
        return str(self.val)

    @test_vec
    def __add__(self, other):
        out_val = self.val + other.val
        return Vector(out_val)

    @test_vec
    def __sub__(self, other):
        out_val = self.val - other.val
        return Vector(out_val)

    @test_num
    def __mul__(self, other):
        return Vector(self.val * other)

    @test_num
    def __rmul__(self, other):
        return Vector(self.val * other)

    @test_num
    def __truediv__(self, other):
        return Vector(self.val / float(other))

    @test_vec
    def Dot(self, vec: "Vector"):
        return np.dot(self.val, vec.val)

    @test_vec
    def Cross(self, vec: "Vector"):
        return np.cross(self.val, vec.val)

    def Norm(self):
        return np.linalg.norm(self.val)

    def Unit_vector(self):
        return Vector(self.val / np.linalg.norm(self.val))

    @test_vec
    def Angle(self, vec: "Vector"):
        return np.arccos(self.Dot(vec) / (self.Norm() * vec.Norm()))

    def Theta(self):
        # missing quadrant check
        if self.dim != 2:
            raise ValueError("Only valid for dim=2")
        else:
            return np.arctan(self.val[0] / self.val[1])

    @test_num
    def Offset(self, val):
        return Vector(self.val + val)

    def Polar(self):
        if self.dim != 2:
            raise ValueError("Only valid for dim=2")
        else:
            return self.Norm(), self.Theta()

    def Polar3D(self):
        pass


if __name__ == "__main__":
    pass
