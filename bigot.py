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


def calculate_complexity(data, function, col, bench_func, plot=False):
    data = data[[col, "n"]].dropna(subset=[col]).groupby(
        ["n"]).median().reset_index()
    data[col] = data[col]-data[col].min()

    # Filtering
    print(data)
    if False:
        window_size = int(data.shape[0]/20)+2
        data_raw = data.copy()
        if col == "t":
            data = data.rolling(window_size).min().dropna(subset=[col])
        elif col == "m":
            data = data.rolling(window_size).max().dropna(subset=[col])

    assert data.shape[0] > 0
    last_chunk = data.quantile([.5, 1]).median()
    first_chunk = data.quantile([0, .5]).median()
    x = list(data.n.unique())
    avg_n = data.n.median()
    avg_t = data[col].median()

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
    pprint(losses)
    for func_name, loss in losses.items():
        if loss < 4:
            func = complex_funcs[func_name]
            t_predicted_further = func(n_further, avg_n, avg_t)
            losses_further[func_name] = t_predicted_further

    if len(losses_further) > 1:
        print("Hesitation", func_closest)

        col_expected_further = bench_func(n_further)  # noqa
        data = data.append({
            "n": n_further,
            col: col_expected_further,
        }, ignore_index=True)
        losses_further = {name: abs(col_further/col_expected_further) for (name, col_further) in losses_further.items()}  # noqa
        func_closest = min(losses_further, key=losses_further.get)
        print("Hesitation", func_closest)
        pprint(losses_further)

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.n, y=data[col], name="Observation",
                                 line=dict(width=1)))
        # fig.add_trace(go.Scatter(x=data_raw.n, y=data_raw[col], name="RAW",
        #                         line=dict(width=1)))

        for func_name, func in complex_funcs.items():
            x = list(data.n.unique())
            y = [func(n, avg_n, avg_t) for n in x]
            if max(y) < 2*data[col].max():
                fig.add_trace(go.Scatter(x=x, y=y, name=func_name,
                                         line=dict(width=1, dash='dot')))

        fig.show()

    return func_closest


def benchmark(function, time_budget=1000, memory=False, plot=False):
    data = pd.DataFrame()

    # Time check
    test_time = time()

    def func_t(n, runs=1): return Timer(functools.partial(
        function, int(n))).timeit(runs)/runs*1000
    for n in [n for n in range(1, 10000)]:
        t = func_t(n)
        data = data.append({
            "n": int(n),
            "t": t
        }, ignore_index=True)
        if (time() - test_time)*1000 > time_budget:
            break
    if data.shape[0] > 10000:
        data = data.sample(n=10000)
    time_complexity = calculate_complexity(data, function, "t", func_t, plot=plot)  # noqa
    print("Time complexity:", time_complexity)


    # Memory check
    if memory:
        def func_m(n): return max(memory_usage(
            proc=(function, [int(n)]), interval=(0.01)))
        for index, row in data.iterrows():
            data.loc[index, "m"] = func_m(row.n)
        space_complexity = calculate_complexity(data, function, "m", func_m, plot=plot)  # noqa
        print("Space complexity:", space_complexity)
