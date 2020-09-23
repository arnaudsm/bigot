from bigot import Bigot
from time import sleep


def myfunction(n):
    x = 10000000*"-"*int(n**2)
    sleep(0.001*n**2)


# First of all create your Bigot object with your function inside
bigot = Bigot(myfunction)

# Then use the space() and time() functions to estimate the complexity
print("My function has a space complexity of", bigot.space(),
      "and a time complexity of", bigot.time())

# You can also plot your function to study its behavior
print(bigot.time(plot=True, time_budget=5))

# You can also calculate the two at once using
print(bigot.all())
