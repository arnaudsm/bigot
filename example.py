import bigot
from time import sleep


def on(n):
    x = 10000000*"-"*int(n)  # noqa
    sleep(0.001*n)


def on2(n):
    x = 10000000*"-"*int(n**2)  # noqa
    sleep(0.001*n**2)


# Use the Space() and Time() classes to benchmark functions
print("My function has a space complexity of", bigot.Space(on2),
      "and a time complexity of", bigot.Time(on2))

# You can test our fancy options
bench = bigot.Time(
    on2,
    plot=True,
    duration=1,
    verbose=True,
    name="My fancy function"
)

# And check the number of iterations, useful when comparing functions
print(bench.iterations, "iterations in", bench.duration, "seconds")

# You can also compare multiple functions
print(bigot.Compare([on, on2]).all())
