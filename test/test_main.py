import time
from math import factorial, log

import pytest

from bigot import Bigot, complex_funcs


def test_time_benchmark():
    def fake_function(n, func):
        time.sleep(0.01*func(n, 2, 2))

    for name, real_function in complex_funcs.items():
        def function(n): return fake_function(n, real_function)
        print("Checking time complexity of", name)
        assert name == Bigot(function, name=name).time()


def test_space_benchmark():
    def fake_function(n, func):
        x = 10000000*"-"*int(func(n, 2, 2))
        time.sleep(0.01)

    for name, real_function in complex_funcs.items():
        def function(n): return fake_function(n, real_function)
        print("Checking space complexity of", name)
        assert name == Bigot(function, name=name).space()
