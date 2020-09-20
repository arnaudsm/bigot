import collections
import math
from math import log
from time import time
import operator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import functools
import timeit

complexities = {
    "O(n)": lambda n, avg_n, avg_t: n*avg_t/avg_n,
    "O(n²)": lambda n, avg_n, avg_t: (avg_t/avg_n**2)*n**2,
    # "O(n³)": lambda n, avg_n, avg_t: (avg_t/avg_n**3)*n**3,
    "O(√n)": lambda n, avg_n, avg_t: (avg_t/avg_n**0.5)*n**0.5,
    "O(n log n)": lambda n, avg_n, avg_t: avg_t / (avg_n*log(avg_n, 2))*n*log(n, 2),
    "O(n² log n)": lambda n, avg_n, avg_t: avg_t / (avg_n**2*log(avg_n, 2))*n**2*log(n, 2),
    "O(log n)": lambda n, avg_n, avg_t: avg_t/(log(avg_n, 2))*log(n, 2),
    # "O(2^n)": lambda n, avg_n, avg_t: 2**n*avg_t/(2**avg_n),
    # "O(n!)": lambda n, avg_n, avg_t: avg_t/(log(avg_n, 2))*log(n, 2),
}


def calculate_complexity(data):
    last_chunk = data.quantile([.5, 1]).median()
    first_chunk = data.quantile([0, .5]).median()
    complexity_exponent = (log(first_chunk.t, 10)-log(last_chunk.t, 10)) / (
        log(first_chunk.n, 10)-log(last_chunk.n, 10))
    complexity_exponent = round(complexity_exponent, 2)

    x = list(data.n.unique())
    avg_n = data.n.median()
    avg_t = data.t.median()

    losses = {}
    for func_name, func in complexities.items():
        y = [func(n, avg_n, avg_t) for n in x]
        loss = (data.t-y).abs().mean()
        losses[func_name] = loss

    sorted_loss = sorted({name: -log(loss/max(losses.values()))
                          for (name, loss) in losses.items()}.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_loss[0:2])

    time_complexity = min(losses, key=losses.get)
    return time_complexity, complexity_exponent


def benchmark(function, time_budget=1000, memory_budget=1000, plot=False):
    data = pd.DataFrame()
    test_time = time()
    for n in [n for n in range(1, 10000)]:
        start_time = time()
        _ = function(n)
        t = (time() - start_time)*1000
        data = data.append({
            "n": int(n),
            "t": t
        }, ignore_index=True)
        if (time() - test_time)*1000 > time_budget:
            break

    max_n = int(data.n.max())
    if data.shape[0] > 10000:
        data = data.sample(n=10000)
    data = data.groupby(["n"]).t.median().to_frame(
        name="t").reset_index().sort_values(by="n")

    window_size = 10
    data_filtered = data.rolling(window_size).min().dropna(how='any')

    time_complexity, complexity_exponent = calculate_complexity(data_filtered)

    print("Time complexity:", time_complexity)
    print("Complexity exponent:", complexity_exponent)
    print("Max N:", max_n)

    if plot:
        data = data.groupby(["n"]).t.median().to_frame(
            name="t").reset_index()
        fig = px.line(data, y="t", x="n")
        fig.add_trace(go.Scatter(x=data_filtered.n, y=data_filtered.t, name="filtered",
                                 line=dict(width=1)))

        x = list(data.n)
        avg_n = data.n.median()
        avg_t = data.t.median()

        for func_name, func in complexities.items():
            y = [func(n, avg_n, avg_t) for n in x]
            fig.add_trace(go.Scatter(x=x, y=y, name=func_name,
                                     line=dict(width=1, dash='dot')))

        fig.show()
