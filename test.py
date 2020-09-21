import bigot
import time
from math import log
from pprint import pprint
from bigot import complex_funcs
time_budget = 1


def test_space():
    for name, function in complex_funcs.items():
        results = bigot.benchmark(
            lambda n: 100000*"-"*int(function(n, 1, 1)),
            time_budget=time_budget)
        print(name, results['Space complexity:'])
        if not results:
            continue
        assert results['Space complexity:'] == name


def test_time():
    for name, function in complex_funcs.items():
        results = bigot.benchmark(
            lambda n: time.sleep(0.01*int(function(n, 1, 1))),
            time_budget=time_budget)
        print(name, results['Time complexity:'])
        if not results:
            continue
        assert results['Time complexity:'] == name


test_time()
# Todo
# Stop when too slow : at least 2 runs
# Test slow functions
# Autotests
