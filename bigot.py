import collections
import math
from math import log
from time import time
import operator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

complexities = {
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


def calculate_complexity(data):
    first_chunk = data.quantile([.5, 1]).median()
    last_chunk = data.quantile([0, .5]).median()
    complexity_exponent = (log(last_chunk.t, 10)-log(first_chunk.t, 10)) / (
        log(last_chunk.n, 10)-log(first_chunk.n, 10))
    complexity_exponent = round(complexity_exponent, 2)

    x = list(data.n.unique())
    avg_n = data.n.median()
    avg_t = data.t.median()

    losses = {}
    for func_name, func in complexities.items():
        y = [func(n, avg_n, avg_t) for n in x]
        loss = (data.t-y).abs().mean()
        losses[func_name] = loss

    losses = {name: loss/max(losses.values())
              for (name, loss) in losses.items()}

    print(sorted(losses.items(), key=operator.itemgetter(1)))
    print(losses)

    time_complexity = min(losses, key=losses.get)
    return time_complexity, complexity_exponent


def benchmark(functions, time_budget=1000, plot=False):
    if isinstance(functions, collections.Callable):
        functions = [functions]

    df = pd.DataFrame()
    for function in functions:
        test_time = time()
        for n in [int(n**2) for n in range(1, 999)]:
            start_time = time()
            _ = function(n)
            t = (time() - start_time)*1000
            df = df.append({
                "name": function.__name__.replace("_", " ").capitalize(),
                "n": int(n),
                "t": t
            }, ignore_index=True)
            if (time() - test_time)*1000*len(functions) > time_budget:
                break
    podium = df.groupby(["name"]).n.max().astype(
        int).sort_values(ascending=False).to_frame(name="Speed")
    print("Simulation done")
    for name, speed in podium.iterrows():
        data = df[df.name == name].sort_values(by="n")

        time_complexity, complexity_exponent = calculate_complexity(data)
        podium.loc[name, "Time complexity"] = time_complexity
        podium.loc[name, "Complexity exponent"] = complexity_exponent
    print("Complexity done")
    if plot:
        ts = df.groupby(["n", "name"]).t.median().to_frame(
            name="t").reset_index()
        fig = px.line(ts, color="name", y="t", x="n")

        if len(functions) == 1:
            x = list(df.n.unique())
            avg_n = df.n.median()
            avg_t = df.t.median()

            for func_name, func in complexities.items():
                y = [func(n, avg_n, avg_t) for n in x]
                fig.add_trace(go.Scatter(x=x, y=y, name=func_name,
                                         line=dict(width=1, dash='dot')))

        fig.show()

    return podium
