import collections
import functools
import math
import operator
from timeit import Timer
from math import log
from time import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from memory_profiler import memory_usage
from pprint import pprint

complex_funcs = {
    "O(n)": lambda n, avg_n, avg_dim: n*avg_dim/avg_n,
    "O(n²)": lambda n, avg_n, avg_dim: (avg_dim/avg_n**2)*n**2,
    "O(n³)": lambda n, avg_n, avg_dim: (avg_dim/avg_n**3)*n**3,
    "O(√n)": lambda n, avg_n, avg_dim: (avg_dim/avg_n**0.5)*n**0.5,
    "O(n log n)": lambda n, avg_n, avg_dim: avg_dim / (avg_n*log(avg_n, 2))*n*log(n, 2),
    "O(n² log n)": lambda n, avg_n, avg_dim: avg_dim / (avg_n**2*log(avg_n, 2))*n**2*log(n, 2),
    "O(log n)": lambda n, avg_n, avg_dim: avg_dim/(log(avg_n, 2))*log(n, 2),
    # "O(2^n)": lambda n, avg_n, avg_dim: 2**n*avg_dim/(2**avg_n),
    # "O(n!)": lambda n, avg_n, avg_dim: avg_dim/(log(avg_n, 2))*log(n, 2),
}

dimensions = {
    "m": {
        "name": "Space complexity:",
        "benchmark": lambda n, func:  max(memory_usage(proc=(func, [int(n)]), interval=(0.01)))  # noqa
    },
    "t": {
        "name": "Time complexity:",
        "benchmark": lambda n, func:  Timer(functools.partial(func, int(n))).timeit(1)  # noqa
    }
}


def calc_complexity(data, dim, plot=False):
    data = data[[dim, "n"]].dropna(subset=[dim]).groupby(
        ["n"]).median().reset_index()
    data[dim] = data[dim]-data[dim].min()

    x = list(data.n.unique())
    avg_n = data.n.median()
    avg_dim = data[dim].median()

    losses = {}
    for func_name, func in complex_funcs.items():
        y = [func(n, avg_n, avg_dim) for n in x]
        loss = (data[dim]-y).abs().mean()
        losses[func_name] = loss

    func_closest = min(losses, key=losses.get)

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.n, y=data[dim], name="Observation",
                                 line=dict(width=1)))

        for func_name, func in complex_funcs.items():
            y = [func(n, avg_n, avg_dim) for n in x]
            if max(y) < 2*data[dim].max():
                fig.add_trace(go.Scatter(x=x, y=y, name=func_name,
                                         line=dict(width=1, dash='dot')))

        fig.show()

    return func_closest


def benchmark(function, time_budget=5, plot=False):

    results = {}
    data = pd.DataFrame()
    for dim, params in dimensions.items():
        name = params["name"]
        benchmark = params["benchmark"]
        test_time = time()
        for n in [int(n**2) for n in range(1, 10000)]:
            t = benchmark(n, function)
            data = data.append({
                "n": int(n),
                dim: t
            }, ignore_index=True)
            if (time() - test_time) > time_budget/len(dimensions):
                break
        if data.shape[0] == 0:
            return False
        complexity = calc_complexity(data, dim, plot=plot)
        results[name] = complexity
    results["Speed (iterations in {}s)".format(
        time_budget)] = int(data.n.max())

    return results
