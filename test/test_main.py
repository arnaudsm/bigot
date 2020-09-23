from time import sleep
from math import factorial, log

import pytest

from bigot import Time, Space, Compare, complex_funcs


def test_time_benchmark():
    def fake_function(n, func):
        sleep(0.01*func(n, 2, 2))

    for name, real_function in complex_funcs.items():
        def function(n): return fake_function(n, real_function)
        print("Checking time complexity of", name)
        assert name == str(Time(function, name=name))


def test_space_benchmark():
    def fake_function(n, func):
        x = 10000000*"-"*int(func(n, 2, 2))
        sleep(0.01)

    for name, real_function in complex_funcs.items():
        def function(n): return fake_function(n, real_function)
        print("Checking space complexity of", name)
        assert name == str(Space(function, name=name))


def test_space_benchmark():
    def on(n):
        x = 10000000*"-"*int(n)
        sleep(0.001*n)

    def on2(n):
        x = 10000000*"-"*int(n**2)
        sleep(0.001*n**2)

    assert Compare([on, on2]).all().shape == (2, 7)
    assert Compare([on, on2]).time().shape == (2, 4)
    assert Compare([on, on2]).space().shape == (2, 4)
