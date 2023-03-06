"""
The robot's operating system is already in place, all joints are properly oiled up, and it is eager to start working.
The module makes sure it does not overextend its maximum load while still being the most effective!
"""


import itertools
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def solve(values: np.ndarray, weights: np.ndarray, maximum_weight: int) -> int:
    """
    :param values: List of values.
    :param weights: List of weights.
    :param maximum_weight: Maximum weight the robot system can handle.
    :return: The optimized maximum weight.
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


def maximum_value(orders: dict, maximum_weight: int) -> int:
    """
    For a given list of orders and a maximum_load, tries to find the best combination of items based on their value.
    :param orders: List of orders, each order is a dictionary, e.g. orders = [{'weight': 5, 'value': 10}].
    :param maximum_weight: Maximum weight the robot system can handle.
    :return: The optimized maximum weight.
    """
    df = pd.DataFrame.from_dict(orders)
    if df.empty:
        return 0
    return solve(np.array(df['value']), np.array(df['weight']), maximum_weight)
