############################################
# some useful modules and functions
# Maurizio Monti 2021
###############################################

import numpy as np
import functools  # stuff for making decorators
import time as tm
import smtplib as sm  # email protocol
import ssl  # ssl protocol
import email.message  # Email formatting tools (e.g. subject, sender, etc.)
import typing as tp  # support for annotations


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


if __name__ == "__main__":
    pass
