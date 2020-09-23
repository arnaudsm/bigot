import collections
import functools
import json
import math
import operator
from math import log
from time import time
from timeit import Timer

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from memory_profiler import memory_usage

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
    "space": lambda n, func:  max(memory_usage(proc=(func, [int(n)]), interval=(0.001))),
    "time": lambda n, func:  Timer(functools.partial(func, int(n))).timeit(1)
}


class Bigot():
    def __init__(self, function, name=None):
        self.function = function
        self.name = name or function.__name__.capitalize().replace("_", " ")
        self.data = pd.DataFrame()
        self.results = {}

    def time(self, *args, **kwargs):
        return self.complexity("time", *args, **kwargs)

    def space(self, *args, **kwargs):
        return self.complexity("space", *args, **kwargs)

    def all(self, *args, **kwargs):
        self.time(*args, **kwargs)
        self.space(*args, **kwargs)
        return self.results

    def benchmark(self, dim, time_budget=5, *args, **kwargs):
        dim_bench = dimensions[dim]
        test_time = time()
        for n in [int(n**2) for n in range(2, 10000)]:
            y = dim_bench(n, self.function)
            self.data = self.data.append({
                "n": int(n),
                dim: y
            }, ignore_index=True)
            if (time() - test_time) > time_budget:
                break

    def complexity(self, dim, plot=False, *args, **kwargs):
        self.benchmark(dim, *args, **kwargs)

        data = self.data[[dim, "n"]].dropna(subset=[dim]).groupby(
            ["n"]).median().reset_index().sort_values(by="n")
        if dim == "space":
            data[dim] = data[dim]-data[dim].min()
            data = data.iloc[1:]

        x = list(data.n.unique())
        avg_n = data.n.max()
        avg_dim = data[dim].max()

        losses = {}
        for name, func in complex_funcs.items():
            y = [func(n, avg_n, avg_dim) for n in x]
            loss = (data[dim]-y).abs().mean()
            losses[name] = loss

        func_closest = min(losses, key=losses.get)

        if plot:
            fig = go.Figure(layout=go.Layout(title=go.layout.Title(
                text="{dim} Complexity of {name} : Probably {func_closest}".format(
                    dim=dim.capitalize(),
                    name=self.name,
                    func_closest=func_closest
                )))
            )
            fig.add_trace(
                go.Scatter(
                    x=data.n, y=data[dim], name="Observation", line=dict(width=1)
                )
            )

            for name, func in complex_funcs.items():
                y = [func(n, avg_n, avg_dim) for n in x]
                if max(y) < 2*data[dim].max():
                    fig.add_trace(go.Scatter(x=x, y=y, name=name,
                                             line=dict(width=1, dash='dot')))

            fig.show()

        self.results[dim] = func_closest
        return func_closest
