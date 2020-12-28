import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from fnc import *


def f(x):
    return (2 ** x) * ((x-1) ** 2) - 2


def f_d1(x):
    return (2 ** x) * (-1 + x) * (2 - np.log(2) + x * np.log(2))


def f_d2(x):
    return (2 ** x) * (2 - 4 * np.log(2) - 2 * x * (-2 + np.log(2)) * np.log(2) + (np.log(2) ** 2) + (x ** 2) * (np.log(2)**2))

a = 5

F = [f, f_d1, f_d2]

print(
    '   E(H_p)\t|  E_der(H_p)\t|  E_der2(H_p)\t|   E(H_Ch)\t|  E_der(H_Ch)\t|  E_der2(H_Ch) |'
)
print('-' * 97)
x = np.linspace(-a, a, num=200)
X = np.linspace(-a, a, num=3)
print(float("{0:.10f}".format(E_Hermite(x, X, F))), '\t| ', end='')
print(float("{0:.10f}".format(E_Hermite_derivative(x, X, F))), '\t| ', end='')
print(float("{0:.10f}".format(E_Hermite_derivative2(x, X, F))), '\t| ', end='')
X_Ch = cheb(3, a)
print(float("{0:.10f}".format(E_Hermite(x, X_Ch, F))), '\t| ', end='')
print(float("{0:.10f}".format(E_Hermite_derivative(x, X_Ch, F))), '\t| ', end='')
print(float("{0:.10f}".format(E_Hermite_derivative2(x, X_Ch, F))), '\t|')

X = np.linspace(-a, a, num=3)
x = np.linspace(-a, a, num=200)
X_Ch = cheb(3, a)
plt.plot(x, f(x), 'r', label='f(x)')
plt.plot(x, Hermite(x, X, F), 'b', label='H_p(x)')
plt.plot(x, Hermite(x, X_Ch, F), 'g', label='H_Ch(x)')
plt.legend()
plt.show()

# 2nd der
print(' E_der(H_p)\t|  E_der(L_p)\t|  E_der(H_Ch)\t|  E_der(L_Ch)\t|')
print('-' * 65)
N = 3
X = np.linspace(-a, a, num=N)
print(float("{0:.9f}".format(E_Hermite_derivative2(x, X, F))),
      '\t| ',
      end='')
print(float("{0:.9f}".format(E_L_derivative2(x, X, F))), '\t| ', end='')
X_Ch = cheb(N, a)
print(float("{0:.9f}".format(E_Hermite_derivative2(x, X_Ch, F))),
      '\t| ',
      end='')
print(float("{0:.9f}".format(E_L_derivative2(x, np.array(X_Ch), F))), '\t|')


print(
    'deg\t|  E_der2(H_p)\t|  E_der2(L_p)\t|  E_der2(H_Ch)\t|  E_der2(L_Ch)\t|')
print('-' * 73)
N = 3
print(N * 2 - 1, '\t|', end=' ')
X = np.linspace(-a, a, num=N)
print(float("{0:.10f}".format(E_Hermite_derivative2(x, X, F))), '\t| ', end='')
X = np.linspace(-a, a, num=N * 2 - 1)
print(float("{0:.10f}".format(E_L_derivative2(x, X, F))), '\t| ', end='')
X_Ch = cheb(N, a)
print(float("{0:.10f}".format(E_Hermite_derivative2(x, X_Ch, F))),
      '\t| ',
      end='')
X_Ch = cheb(N * 2 - 1, a)
print(float("{0:.10f}".format(E_L_derivative2(x, np.array(X_Ch), F))), '\t|')