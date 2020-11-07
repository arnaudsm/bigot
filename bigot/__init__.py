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


class Benchmark():
    def __init__(self, function, name=None, verbose=False, duration=5, **kwargs):
        """Abstract class to benchmark a function

        Args:
            function (function): The function to benchmark, with a single n parameter.
            name (string, optional): Your function name, only used in graphics. Defaults to the function name.
            verbose (bool, optional): Display all logs. Defaults to False.
            duration (int, optional): Maximum time allocated in seconds. Defaults to 5.
            plot (bool, optional): Wether to display a plotly chart comparing your complexity to usual classes.
        """
        self.function = function
        self.name = name or function.__name__.capitalize().replace("_", " ")
        self.data = pd.DataFrame()
        self.verbose = verbose
        self.duration = duration
        self.complexity = self.get_complexity(**kwargs)

    def __repr__(self):
        return self.complexity

    @property
    def iterations(self):
        return int(self.data.n.max())

    def benchmark(self, **kwargs):
        test_time = time()
        self.log("Benchmarking {}: ".format(self.dim), end="")
        for n in list(range(1, 9))+[int(n**2) for n in range(3, 10000)]:
            self.data = self.data.append({
                "n": int(n),
                self.dim: self.bench_func(n, self.function),
                "name": self.name,
            }, ignore_index=True)
            if (time() - test_time) > self.duration:
                break

        self.log("Done {} iterations".format(n))

    def get_complexity(self, plot=False, **kwargs):
        # Benchmark the function first
        try:
            self.benchmark(**kwargs)
        except Exception as e:
            # Ignore memory errors if enough points were collected
            if type(e).__name__ != "MemoryError" or self.data.shape[0] < 3:
                self.log(e)
                return

        # Filter and sort benchmark data
        data = self.data[[self.dim, "n"]].dropna(subset=[self.dim]).groupby(
            ["n"]).median().reset_index().sort_values(by="n")

        # Zeroing minimal memory
        if self.dim == "space":
            data[self.dim] = data[self.dim]-data[self.dim].min()
            data = data.iloc[1:]

        # Remove excessive early points for longer benchmarks
        if data.n.max() > 30:
            data = data[~data.n.isin([2, 3, 4, 6, 7, 8])]

        x = list(data.n.unique())
        max_n = data.n.max()
        max_dim = data[self.dim].max()

        losses = {}
        for name, func in complex_funcs.items():
            if max_n > 30 and name in explosive_funcs:
                continue
            y = [func(n, max_n, max_dim) for n in x]
            loss = (data[self.dim]-y).abs().mean()
            losses[name] = loss

        func_closest = min(losses, key=losses.get)

        if plot:
            import plotly.graph_objects as go
            fig = go.Figure(layout=go.Layout(title=go.layout.Title(
                text="{dim} Complexity of {name} : Probably {func_closest}".format(
                    dim=self.dim.capitalize(),
                    name=self.name,
                    func_closest=func_closest
                )))
            )
            fig.add_trace(
                go.Scatter(
                    x=data.n, y=data[self.dim], name="Observation", line=dict(width=2, color='black')
                )
            )

            for name, func in complex_funcs.items():
                if name in explosive_funcs and func_closest not in explosive_funcs:
                    continue
                y = [func(n, max_n, max_dim) for n in x]
                if max(y) < 2*data[self.dim].max():
                    fig.add_trace(go.Scatter(x=x, y=y, name=name,
                                             line=dict(width=1, dash='dot')))

            fig.show()

        return func_closest

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class Time(Benchmark):
    def __init__(self, function,  **kwargs):
        """Benchmark the time complexity of a function

        Args:
            function (function): The function to benchmark, with a single n parameter.
            name (string, optional): Your function name, only used in graphics. Defaults to the function name.
            verbose (bool, optional): Display all logs. Defaults to False.
            duration (int, optional): Maximum time allocated in seconds. Defaults to 5.
            plot (bool, optional): Wether to display a plotly chart comparing your complexity to usual classes.
        """
        self.bench_func = lambda n, func:  Timer(
            functools.partial(func, int(n))).timeit(1)
        self.dim = "time"
        super().__init__(function,  **kwargs)


class Space(Benchmark):
    def __init__(self, function,  **kwargs):
        """Benchmark the space complexity of a function

        Args:
            function (function): The function to benchmark, with a single n parameter.
            name (string, optional): Your function name, only used in graphics. Defaults to the function name.
            verbose (bool, optional): Display all logs. Defaults to False.
            duration (int, optional): Maximum time allocated in seconds. Defaults to 5.
            plot (bool, optional): Wether to display a plotly chart comparing your complexity to usual classes.
        """
        self.bench_func = lambda n, func:  max(
            memory_usage(proc=(func, [int(n)]), interval=(0.001)))
        self.dim = "space"
        super().__init__(function, **kwargs)


class Compare():
    def __init__(self, functions, **kwargs):
        """Benchmark the complexity of multiple functions

        Args:
            functions (array): The functions to benchmark, each with a single n parameter.
            verbose (bool, optional): Display all logs. Defaults to False.
            duration (int, optional): Maximum time allocated in seconds. Defaults to 5.
        """
        self.functions = functions
        self.results = {}

    def run(self, dim, **kwargs):
        assert dim in ["time", "space"]
        plot = kwargs.get("plot")
        if plot:
            del kwargs["plot"]

        data = pd.DataFrame()
        for function in self.functions:
            result = globals()[dim.capitalize()](function, **kwargs)

            self.results[result.name] = {
                **self.results.get(result.name, {}),
                **{
                    "{} iterations".format(dim.capitalize()): result.iterations,
                    "{} complexity".format(dim.capitalize()): result.complexity,
                }
            }

            data = data.append(result.data)

        if plot:
            import plotly.express as px
            px.line(
                data,
                x="n",
                y=dim,
                color="name",
                title="{} complexities".format(dim.capitalize())
            ).show()

        return pd.DataFrame.from_dict(self.results, orient="index")

    def all(self, **kwargs):
        self.run("space", **kwargs)
        self.run("time", **kwargs)

        return pd.DataFrame.from_dict(self.results, orient="index")

    def time(self, **kwargs):
        self.run("time", **kwargs)
        return pd.DataFrame.from_dict(self.results, orient="index")

    def space(self, **kwargs):
        self.run("space", **kwargs)
        return pd.DataFrame.from_dict(self.results, orient="index")
