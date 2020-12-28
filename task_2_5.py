import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from fnc import *

def f(x):
    return (2 ** x) * ((x-1) ** 2) - 2

def f_1(x):
    return (2 ** x) * (-1 + x) * (2 - np.log(2) + x * np.log(2))

a = 5
n = 4
x = np.linspace(-a, a, num=200)
X = np.linspace(-a, a, num=n)

plt.plot(x, f(x), 'r', label='f(x)')
plt.plot(x, [S31(_, X, f, f_1) for _ in x], 'b', label='S_10(x)')
plt.legend()
plt.show()

n = 4
x = np.linspace(-a, a, num=200)
X = np.linspace(-a, a, num=n)


plt.plot(x, f(x), 'r', label='f(x)')
plt.plot(x, [S32_natural(_, X, f) for _ in x], 'b', label='S_32_natural(x)')
plt.legend()
plt.show()

n = 3
x = np.linspace(-a, a, num=200)
X = np.linspace(-a, a, num=n)

plt.rcParams["figure.figsize"] = (12, 12)
fig = plt.figure()
plt.plot(x, f(x), 'r', label='f(x)')
plt.plot(x, [S10(_, X,f) for _ in x], 'g', label='S_10(x)')
plt.plot(x, [S31(_, X, f, f_1) for _ in x], 'b', label='S_31(x)')
plt.plot(x, [S32_der(_, X, f, f_1) for _ in x], 'y', label='S_32_der(x)')
plt.plot(x, [S32_natural(_, X, f) for _ in x], 'm', label='S_32_natural(x)')
plt.legend()
plt.show()
