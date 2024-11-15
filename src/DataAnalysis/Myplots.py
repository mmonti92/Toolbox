import matplotlib.pyplot as plt
from typing import Any


class PrettyAxes(plt.Axes):
    """docstring for PrettyAxes"""

    def __init__(
        self,
        fig: plt.Figure,
        rect: list[float, float, float, float] = [0.1, 0.1, 0.85, 0.85],
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(fig, rect, *args, **kwargs)

    def plot(
        self,
        *args: Any,
        marker: str = "o",
        mfc: str = "None",
        ls: str = "",
        xlabel: str = "",
        ylabel: str = "",
        xlim: list[float, float] = [None, None],
        ylim: list[float, float] = [None, None],
        **kwargs: Any
    ) -> plt.Line2D:
        p = super().plot(*args, **kwargs, marker=marker, mfc=mfc, ls=ls)
        self.set_ylim(ylim)
        self.set_xlim(xlim)
        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        self.tick_params(
            which="major",
            length=10,
            direction="in",
            right=True,
            top=True,
        )
        return p


def PrettyPlot(*args: Any, **kwargs: Any) -> list[plt.Line2D]:
    fig = plt.figure()
    axpp = PrettyAxes(fig)
    ax = fig.add_axes(axpp)
    out = ax.plot(*args, **kwargs)
    ax.legend()
    fig.tight_layout()
    return out


def PrettyFrame(*args: Any, **kwargs: Any) -> tuple[plt.Figure, PrettyAxes]:
    fig = plt.figure()
    axpp = PrettyAxes(fig)
    ax = fig.add_axes(axpp)
    return fig, ax


if __name__ == "__main__":
    pass
