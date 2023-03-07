"""
# Profiler and timing decorator to analyse the code performance
"""

import functools
import time

from pyinstrument import Profiler


def code_profiler(function):
    """
    Decorator to analyze the steps made by the code in a function.
    :param function:
    :return:
    """
    @functools.wraps(function)
    def wrapper_decorator(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        # start the function to profile
        value = function(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True, show_all=True))
        return value
    return wrapper_decorator


def timeit(function):
    """
    Decorator to check the consumed time in a function.
    :param function:
    :return:
    """
    def timed(*args, **kw):
        start = time.time()
        result = function(*args, **kw)
        end = time.time()
        print('%r  %2.2f sec' % (function.__name__, (end - start)))
        return result
    return timed
