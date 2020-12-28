import numpy as np
import matplotlib.pyplot as plt
import random
from func3 import *


a = 1  # границы отрезка
X = np.linspace(-a, a, num=3)
x = np.linspace(-a, a, num=200)
X_Ch = Ch(3, a)


plt.plot(x, [f(_) for _ in x], 'r', label='f(x)')
plt.plot(x, [PolynomialDegN(_, X, f) for _ in x], 'b', label='3d degree equidistant')
plt.plot(x, [PolynomialDegN(_, X_Ch, f) for _ in x], 'g.', label='3d degree Chebyshev')
plt.legend()
plt.show()


x = np.linspace(-a, a, num=200)
print('N\t| E(eq nodes)\t|  E(Ch nodes)\t|')
print('-'*41)
for N in range(3,11):
    print(N,'\t|', end=' ')
    X = np.linspace(-a, a, num = N)
    print(float("{0:.10f}".format(error(x, X, f,PolynomialDegN))),'\t|', end='')
    X_Ch = Ch(N, a)
    print(float("{0:.10f}".format(error(x, X_Ch, f,PolynomialDegN))),'\t|')


x = np.linspace(-a, a, num=200)
print('\t|\t3d degree p\t|\tLagrange p\t|')
print('-'*73)
print('N\t|' + ' E(eq nodes)\t|  E(Ch nodes)\t|'*2)
print('-'*73)
for N in range(3,15):
    print(N,'\t|', end=' ')
    X = np.linspace(-a, a, num = N)
    X_Ch = Ch(N, a)
    print(float("{0:.10f}".format(error(x, X, funcWithError, PolynomialDegN))),'\t| ', end='')
    print(float("{0:.10f}".format(error(x, X_Ch, funcWithError, PolynomialDegN))),'\t| ', end='')
    print(float("{0:.10f}".format(error(x, X, funcWithError, Lagrange))),'\t| ', end='')
    print(float("{0:.10f}".format(error(x, X_Ch, funcWithError, Lagrange))),'\t|')

X = np.linspace(-a, a, num=3)
x = np.linspace(-a, a, num=200)
plt.plot(x, [f(_) for _ in x], 'r', label='f(x)')
plt.plot(x, [PolynomialDegN(_, X, funcWithError) for _ in x], 'b', label='3d degree equidistant nodes')
plt.plot(x, [Lagrange(_, X, funcWithError) for _ in x], 'g', label='Lagrange equidistant nodes')
plt.legend()
plt.show()


x = np.linspace(-a, a, num=200)
print('N\t| E(3d degree with e)\t| E(3d degree ) |')
print('-'*50)
for N in range(3,40):
    print(N,'\t|', end=' ')
    X = np.linspace(-a, a, num = N)
    print(float("{0:.10f}".format(error(x, X, funcWithError, PolynomialDegN))),'\t\t| ', end='')
    print(float("{0:.10f}".format(error(x, X, f, PolynomialDegN))),'\t|')
    print('\n','E(Legendre):',float("{0:.10f}".format(error(x, X, f, Legendre))))

X = np.linspace(-a, a, num=3)
x = np.linspace(-a, a, num=200)
plt.plot(x, [f(_) for _ in x], 'r', label='f(x)')
plt.plot(x, [PolynomialDegN(_, X, f) for _ in x], 'b', label='3d degree equidistant nodes')
plt.plot(x, [Legendre(_, X, f) for _ in x], 'g', label='Legendre')

plt.legend()
plt.show()