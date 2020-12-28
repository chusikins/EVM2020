import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from fnc import *


def orig_f(x):
    return (2 ** x) * ((x-1) ** 2) - 2


def f(x):
    return orig_f(x) * abs(x)


a = 5  # границы
n = 6  # узлы
x = np.linspace(-a, a, num=200)
X = np.linspace(-a, a, num=n)
X_cheb = cheb(n, a)
print('-'*40)
print("1.1 График")
plt.plot(x, [f(_) for _ in x], 'r', label='f(x)')
plt.plot(x, Lagrange(x, X, f), 'b', label='L_p(x)')
plt.plot(x, Lagrange(x, X_cheb, f), 'g', label='L_Ch(x)')
plt.legend()
plt.show()
print('-'*40)
print("1.2 и 1.3 Погрешность и Таблица")

print('N\t|    E(L_p)\t|    E(L_Ch)\t|')
print('-'*40)
for N in range(3, 13):
    print(N,'\t|', end=' ')
    X=np.linspace(-a, a, num=N)
    print(float("{0:.10f}".format(E_L(x, X, f))),'\t| ', end='')
    X_cheb = cheb(N, a)
    print(float("{0:.10f}".format(E_L(x, X_cheb, f))), '\t|')

