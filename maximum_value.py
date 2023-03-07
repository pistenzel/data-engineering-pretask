"""
The robot's operating system is already in place, all joints are properly oiled up, and it is eager to start working.
The module makes sure it does not overextend its maximum load while still being the most effective!
"""

import functools
import itertools
import logging

from pyinstrument import Profiler

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def prepare_data(orders):
    """
    Method to prepare the data for the maximum profit calculation. It transforms a list of orders into a list of values
    and a list of weights. The method provides an initial validation of the data input and returns two empty lists in
    error case.
    :param orders:
    :return: List of values and list of weights. Returns empty lists for invalid orders.
    """
    values = []
    weights = []
    try:
        df = pd.DataFrame.from_dict(orders)
        if df.empty and not df.columns.isin(['value', 'weight']).any():
            raise ValueError('DataFrame empty - No input values or weights given!')
        values = df['value'].tolist()
        weights = df['weight'].tolist()
    except ValueError as ex:
        logger.error(f'prepare_data - {ex}')
    return values, weights


def solve(values: np.ndarray, weights: np.ndarray, maximum_weight: int) -> int:
    """
    Method that provides a solution for the known Knapsack problem: https://en.wikipedia.org/wiki/Knapsack_problem
    Based on the weights and values with each n items, and the maximum weight (maximum capacity) the maximum profit is
    calculated.

    :param values: List of values.
    :param weights: List of weights.
    :param maximum_weight: Maximum weight the robot system can handle.
    :return: The maximum profit.
    """
    n = len(values)
    matrix = [[0 for _ in range(maximum_weight + 1)] for _ in range(n + 1)]

    for i, j in itertools.product(range(n + 1), range(maximum_weight + 1)):
        if i == 0 or j == 0:
            matrix[i][j] = 0
        elif weights[i - 1] <= j:
            matrix[i][j] = max(values[i - 1] + matrix[i - 1][j - weights[i - 1]], matrix[i - 1][j])
        else:
            matrix[i][j] = matrix[i - 1][j]
    return matrix[len(values)][maximum_weight]


@code_profiler
def maximum_value(orders: list, maximum_weight: int) -> int:
    """
    For a given list of orders and a maximum_load, tries to find the best combination of items based on their value.
    :param orders: List of orders, each order is a dictionary, e.g. orders = [{'weight': 5, 'value': 10}].
    :param maximum_weight: Maximum weight the robot system can handle.
    :return: The optimized maximum weight.
    """
    values, weights = prepare_data(orders)
    return solve(values, weights, maximum_weight)
