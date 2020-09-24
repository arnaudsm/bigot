![Example complexity graph](https://raw.githubusercontent.com/arnaudsm/bigot/master/images/graph.png)
# Bigot
Benchmarking library with Space and Time Complexity estimation.  
*Pull requests are welcome !*

## Installation
`pip install bigot`

## Usage
### Provide a benchmark function with a single dimension parameter
```python
def on(n):
    x = 10000000*"-"*int(n)
    sleep(0.001*n)
    
import bigot
print("Function has a space complexity of", bigot.Space(on2),
      "and a time complexity of", bigot.Time(on2))
```
```
Function has a space complexity of O(n^2) and a time complexity of O(n^2)
```

### You can test our fancy options. See docstrings for reference.
```python
bench = bigot.Time(
    on2,
    plot=True,
    duration=1,
    verbose=True,
    name="My fancy function"
)
```

### And check the number of iterations, useful when comparing functions
```python
print(bench.iterations, "iterations in", bench.duration, "seconds")
```
```
8 iterations in 8 seconds
```
### You can also compare multiple functions
```python
def on2(n):
    x = 10000000*"-"*int(n**2)
    sleep(0.001*n**2)

print(bigot.Compare([on, on2]).space())
```
```
  Name  Duration  Iterations Space complexity
0   On       1.0        49.0             O(n)
1  On2       1.0         8.0           O(n^2)
```

## Testing
`pytest .`
