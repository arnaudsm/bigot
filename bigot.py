import collections
import functools
import math
import operator
import timeit
from math import log
from time import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from memory_profiler import memory_usage

complex_funcs = {
    "O(n)": lambda n, avg_n, avg_t: n*avg_t/avg_n,
    "O(n²)": lambda n, avg_n, avg_t: (avg_t/avg_n**2)*n**2,
    "O(n³)": lambda n, avg_n, avg_t: (avg_t/avg_n**3)*n**3,
    "O(√n)": lambda n, avg_n, avg_t: (avg_t/avg_n**0.5)*n**0.5,
    "O(n log n)": lambda n, avg_n, avg_t: avg_t / (avg_n*log(avg_n, 2))*n*log(n, 2),
    "O(n² log n)": lambda n, avg_n, avg_t: avg_t / (avg_n**2*log(avg_n, 2))*n**2*log(n, 2),
    "O(log n)": lambda n, avg_n, avg_t: avg_t/(log(avg_n, 2))*log(n, 2),
    # "O(2^n)": lambda n, avg_n, avg_t: 2**n*avg_t/(2**avg_n),
    # "O(n!)": lambda n, avg_n, avg_t: avg_t/(log(avg_n, 2))*log(n, 2),
}


def calculate_complexity(data, function, col, plot=False, filter_values=False):
    data = data[[col, "n"]].dropna(subset=[col])
    if filter_values:
        window_size = int(data.shape[0]/20)+1
        data = data.rolling(window_size).min().dropna(subset=[col])
    print(1)
    print(data)
    print(data.shape)
    print(data.quantile([.5, 1]).median())
    assert data.shape[0] > 0
    last_chunk = data.quantile([.5, 1]).median()
    print(1.5)
    first_chunk = data.quantile([0, .5]).median()
    print(2)
    x = list(data.n.unique())
    avg_n = data.n.median()
    avg_t = data[col].median()
    print(5)

    losses = {}
    for func_name, func in complex_funcs.items():
        y = [func(n, avg_n, avg_t) for n in x]
        loss = (data[col]-y).abs().mean()
        losses[func_name] = loss

    losses = {name: loss/min(losses.values())
              for (name, loss) in losses.items()}

    func_closest = min(losses, key=losses.get)

    # Hesitation check
    losses_further = {}
    n_further = int(data.n.max() * 2)
    for func_name, loss in losses.items():
        if loss < 2:
            func = complex_funcs[func_name]
            t_predicted_further = func(n_further, avg_n, avg_t)
            losses_further[func_name] = t_predicted_further

    if len(losses_further) > 1:
        t_expected_further = timeit.Timer(functools.partial(function, n_further)).timeit(10)/10*1000  # noqa
        losses_further = {name: abs(t_further/t_expected_further) for (name, t_further) in losses_further.items()}  # noqa
        func_closest = min(losses_further, key=losses_further.get)

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.n, y=data[col], name="Observation",
                                 line=dict(width=1)))

        for func_name, func in complex_funcs.items():
            y = [func(n, avg_n, avg_t) for n in x]
            if max(y) < 2*data[col].max():
                fig.add_trace(go.Scatter(x=x, y=y, name=func_name,
                                         line=dict(width=1, dash='dot')))

        fig.show()

    return func_closest


def benchmark(function, time_budget=1000, memory_budget=1000, plot=False):
    data = pd.DataFrame()

    # Time check
    test_time = time()
    time_function = lambda n: timeit.Timer(functools.partial(function, n)).timeit(3)/3*1000
    min_t = timeit.Timer(functools.partial(function, 1)).timeit(10)/10*1000
    for n in [n for n in range(1, 10000)]:
        t = timeit.Timer(functools.partial(function, n)).timeit(3)/3*1000 - min_t
        data = data.append({
            "n": int(n),
            "t": t
        }, ignore_index=True)
        if (time() - test_time)*1000 > time_budget:
            break
    if data.shape[0] > 10000:
        data = data.sample(n=10000)

    # Memory check
    min_m = max(memory_usage(proc=(function, [data.iloc[0].n]), interval=(data.iloc[0].t/1000000)))  # noqa
    for index, row in data.sample(n=10).iterrows():
        data.loc[index, "m"] = max(memory_usage(proc=(function, [row.n]), interval=(row.t/1000000))) - min_m  # noqa

    data = data.groupby(["n"]).median().reset_index().sort_values(by="n")

    time_complexity = calculate_complexity(data, function, "t", filter_values=True)  # noqa
    print("Time complexity:", time_complexity)
    space_complexity = calculate_complexity(data, function, "m", plot=True)  # noqa
    print("Space complexity:", space_complexity)
