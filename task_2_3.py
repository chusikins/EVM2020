import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from fnc import *


def f(x):
    return (2 ** x) * ((x-1) ** 2) - 2


def f_d(x):
    return (2 ** x) * (-1 + x) * (2 - np.log(2) + x * np.log(2))


a = 5
F = [f, f_d]

X = np.linspace(-a, a, num=3)
x = np.linspace(-a, a, num=200)
X_Ch = cheb(3, a)
print('-'*40)
plt.plot(x, f(x), 'r', label='f(x)')
plt.plot(x, [Hermite(_, X, F) for _ in x], 'b', label='H_p(x)')
plt.plot(x, [Hermite(_, X_Ch, F) for _ in x], 'g', label='H_Ch(x)')
plt.legend()
plt.show()

print('-'*100)
print('N\t|    E(H_p)\t|    E_der(H_p)\t|    E(H_Ch)\t|  E_der(H_Ch)  |')
print('-' * 70)
for N in range(3, 6):
    print(N, '\t|', end=' ')
    X = np.linspace(-a, a, num=N)
    print(float("{0:.10f}".format(E_Hermite(x, X, F))), '\t| ', end='')
    print(float("{0:.10f}".format(E_Hermite_derivative(x, X, F))), '\t| ', end='')
    X_Ch = cheb(N, a)
    print(float("{0:.10f}".format(E_Hermite(x, X_Ch, F))), '\t| ', end='')
    print(float("{0:.10f}".format(E_Hermite_derivative(x, X_Ch, F))), '\t|')
print('-'*100)

# сравнение погрешностей приближения функции
print('N\t|    E(H_p)\t|    E(L_p)\t|    E(H_Ch)\t|  E(L_Ch)\t|')
print('-' * 70)
for N in range(3, 6):
    print(N, '\t|', end=' ')
    X = np.linspace(-a, a, num=N)
    print(float("{0:.10f}".format(E_Hermite(x, X, F))), '\t| ', end='')
    print(float("{0:.10f}".format(E_L(x, X, f))), '\t| ', end='')
    X_Ch = cheb(N, a)
    print(float("{0:.10f}".format(E_Hermite(x, X_Ch, F))), '\t| ', end='')
    print(float("{0:.10f}".format(E_L(x, X_Ch, f))), '\t|')
print('-'*100)

# сравнение погрешностей приближения производной функции
print('N\t|  E_der(H_p)\t|  E_der(L_p)\t|  E_der(H_Ch)\t|  E_der(L_Ch)\t|')
print('-' * 73)
for N in range(3, 6):
    print(N, '\t|', end=' ')
    X = np.linspace(-a, a, num=N)
    print(float("{0:.9f}".format(E_Hermite_derivative(x, X, F))),
          '\t| ',
          end='')
    print(float("{0:.9f}".format(E_L_derivative(x, X, F))), '\t| ', end='')
    X_Ch = cheb(N, a)
    print(float("{0:.9f}".format(E_Hermite_derivative(x, X_Ch, F))),
          '\t| ',
          end='')
    print(float("{0:.9f}".format(E_L_derivative(x, X_Ch, F))), '\t|')
print('-'*100)


X = np.linspace(-a, a, num=5)
x = np.linspace(-a, a, num=200)
plt.plot(x, f(x), 'r', label='f(x)')
plt.plot(x, Hermite(x, X, F), 'b', label='H_p(x)')
plt.plot(x, Lagrange(x, X, f), 'g', label='L_p(x)')
plt.legend()
plt.show()
print('-'*100)


# сравнение погрешностей приближения производной функции
print('deg\t|  E_der(H_p)\t|  E_der(L_p)\t|  E_der(H_Ch)\t|  E_der(L_Ch)\t|')
print('-' * 70)
for N in range(3, 6):
    print(N * 2 - 1, '\t|', end=' ')
    X = np.linspace(-a, a, num=N)
    print(float("{0:.10f}".format(E_Hermite_derivative(x, X, F))),
          '\t| ',
          end='')
    X = np.linspace(-a, a, num=N * 2 - 1)
    print(float("{0:.10f}".format(E_L_derivative(x, X, F))), '\t| ', end='')
    X_Ch = cheb(N, a)
    print(float("{0:.10f}".format(E_Hermite_derivative(x, X_Ch, F))),
          '\t| ',
          end='')
    X_Ch = cheb(N * 2 - 1, a)
    print(float("{0:.10f}".format(E_L_derivative(x, X_Ch, F))), '\t|')
print('-'*100)
x = np.linspace(-a, a, num=200)
plt.plot(x, f(x), 'r', label='f(x)')
X = np.linspace(-a, a, num=3)
plt.plot(x, Hermite(x, X, F), 'b', label='H_p(x)')
X = np.linspace(-a, a, num=5)
plt.plot(x, Lagrange(x, X, f), 'g', label='L_p(x)')
plt.legend()
plt.show()