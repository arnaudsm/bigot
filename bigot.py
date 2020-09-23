import functools
from math import factorial, log
from time import time
from timeit import Timer

import pandas as pd
from memory_profiler import memory_usage

complex_funcs = {
    "O(1)": lambda n, max_n, max_dim: max_dim,
    "O(n)": lambda n, max_n, max_dim: n*max_dim/max_n,
    "O(n^2)": lambda n, max_n, max_dim: (max_dim/max_n**2)*n**2,
    "O(n^3)": lambda n, max_n, max_dim: (max_dim/max_n**3)*n**3,
    "O(√n)": lambda n, max_n, max_dim: (max_dim/max_n**0.5)*n**0.5,
    "O(n log n)": lambda n, max_n, max_dim: max_dim / (max_n*log(max_n, 2))*n*log(n, 2),
    "O(n^2 log n)": lambda n, max_n, max_dim: max_dim / (max_n**2*log(max_n, 2))*n**2*log(n, 2),
    "O(log n)": lambda n, max_n, max_dim: max_dim/(log(max_n, 2))*log(n, 2),
    "O(2^n)": lambda n, max_n, max_dim: 2**n*max_dim/(2**max_n),
    "O(n!)": lambda n, max_n, max_dim: max_dim/factorial(max_n)*factorial(n),
}

explosive_funcs = ["O(2^n)", "O(n!)"]

dimensions = {
    "space": lambda n, func:  max(memory_usage(proc=(func, [int(n)]), interval=(0.001))),
    "time": lambda n, func:  Timer(functools.partial(func, int(n))).timeit(1)
}


class Bigot():
    def __init__(self, function, name=None, verbose=False):
        self.function = function
        self.name = name or function.__name__.capitalize().replace("_", " ")
        self.data = pd.DataFrame()
        self.verbose = verbose
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
        self.log("Benchmarking {}: ".format(dim), end="")
        for n in list(range(1, 9))+[int(n**2) for n in range(3, 10000)]:
            self.data = self.data.append({
                "n": int(n),
                dim: dim_bench(n, self.function)
            }, ignore_index=True)
            if (time() - test_time) > time_budget:
                break

        self.log("Done {} iterations".format(n))

    def complexity(self, dim, plot=False, *args, **kwargs):
        # Benchmark the function first
        try:
            self.benchmark(dim, *args, **kwargs)
        except Exception as e:
            # Ignore memory errors if enough points were collected
            if type(e).__name__ != "MemoryError" or self.data.shape[0] < 3:
                self.log(e)
                return

        # Filter and sort benchmark data
        data = self.data[[dim, "n"]].dropna(subset=[dim]).sort_values(by="n")

        # Zeroing minimal memory
        if dim == "space":
            data[dim] = data[dim]-data[dim].min()
            data = data.iloc[1:]

        # Remove excessive early points for longer benchmarks
        if data.n.max() > 30:
            data = data[~data.n.isin([2, 3, 4, 6, 7, 8])]

        x = list(data.n.unique())
        max_n = data.n.max()
        max_dim = data[dim].max()

        losses = {}
        for name, func in complex_funcs.items():
            if max_n > 30 and name in explosive_funcs:
                continue
            y = [func(n, max_n, max_dim) for n in x]
            loss = (data[dim]-y).abs().mean()
            losses[name] = loss

        func_closest = min(losses, key=losses.get)

        if plot:
            import plotly.graph_objects as go
            fig = go.Figure(layout=go.Layout(title=go.layout.Title(
                text="{dim} Complexity of {name} : Probably {func_closest}".format(
                    dim=dim.capitalize(),
                    name=self.name,
                    func_closest=func_closest
                )))
            )
            fig.add_trace(
                go.Scatter(
                    x=data.n, y=data[dim], name="Observation", line=dict(width=2, color='black')
                )
            )

            for name, func in complex_funcs.items():
                if name in explosive_funcs and func_closest not in explosive_funcs:
                    continue
                y = [func(n, max_n, max_dim) for n in x]
                if max(y) < 2*data[dim].max():
                    fig.add_trace(go.Scatter(x=x, y=y, name=name,
                                             line=dict(width=1, dash='dot')))

            fig.show()

        self.results[dim] = func_closest
        return func_closest

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
