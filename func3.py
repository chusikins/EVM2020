import numpy as np
import matplotlib.pyplot as plt
import random


def f(x):
    return x * np.cos(x+3)


# поиск корней полинома Чебышева
def Ch(n, a):
    X = []
    for i in range(n):
        X.append(a * np.cos((2 * i + 1) / (2 * n) * np.pi))
    return X

# LU разложение
def LU(u, n):
    l = np.zeros((n, n), dtype=float)
    p = np.eye(n)
    q = np.eye(n)
    for i in range(n):
        u1 = u[i:, i:]
        a = u1.argmax()
        col = a % (n - i)
        line = a // (n - i)
        p1 = np.eye(n, dtype=int)
        p1[i, i] = 0
        p1[line + i, line + i] = 0
        p1[line + i, i] = 1
        p1[i, line + i] = 1
        q1 = np.eye(n, dtype=int)
        q1[i, i] = 0
        q1[col + i, col + i] = 0
        q1[i, col + i] = 1
        q1[col + i, i] = 1
        p = p1.dot(p)
        q = q.dot(q1)
        u = (p1.dot(u)).dot(q1)
    for i in range(n):
        l[i, i] = 1
        for j in range(i + 1, n):
            l[j, i] = u[j, i] / u[i, i]
            u[j] -= l[j, i] * u[i]
    return l, u, p, q


def AxB(l, u, b, n):
    y = np.zeros(n, dtype=float)  # Ly=b
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i]
        for k in range(i):
            y[i] -= l[i, k] * y[k]
    nul_lines_U = 0  # проверка на совместность
    nul_y = 0
    for i in reversed(range(n)):
        flag = True
        for j in range(n):
            if abs(u[i, j]) > 1e-10:
                flag = False
        if flag:
            nul_lines_U += 1
        if abs(y[i]) < 1e-10:
            nul_y += 1
    if nul_lines_U > nul_y:
        print('Система не совместна')
    else:
        rang = n - nul_lines_U
        x = np.zeros(n, dtype=float)  # Ux=y
        filled_x = []
        for i in range(n):
            filled_x.append(0)
        for i in reversed(range(rang)):
            p = i
            unknown = 0
            for k in range(p, n):
                if filled_x[k] == 0 and abs(u[i, k]) > 1e-10:
                    unknown += 1
                    if unknown == 1:
                        p = k
            if unknown > 1:
                for k in range(p + 1, n):
                    if filled_x[k] == 0 and abs(u[i, k]) > 1e-10:
                        x[k] = 0
                        filled_x[k] = 1
            x[p] = y[i]
            k = i
            while k != n - 1:
                k += 1
                if k != p and abs(u[i, k]) > 1e-10:
                    x[p] -= u[i, k] * x[k]
            x[p] = x[p] / u[i, p]
            filled_x[p] = 1
        for i in range(n):
            if filled_x[i] == 0:
                x[i] = 0
        return x
    return 0


def PolynomialDegN(x, X, f, deg=3):
    numOfNodes = len(X)
    numOfCoeff = deg + 1

    Q = np.zeros((numOfNodes, 4))
    y = np.zeros((numOfNodes, 1))
    for i in range(numOfNodes):
        for j in range(numOfCoeff):
            Q[i][j] = X[i]**j
        y[i] = f(X[i])
    H = Q.T @ Q
    b = Q.T @ y
    l, u, p, q = LU(H, numOfCoeff)
    a = AxB(l, u, p.dot(b), numOfCoeff)
    a = q @ a
    result = 0
    for i in range(numOfCoeff):
        result += a[i] * x**i
    return result

def error(x, X, func, polynomial, f=f):
    sum = 0
    for i in x:
        diff = abs(f(i) - polynomial(i, X, func))
        sum += diff**2
    return np.sqrt(sum / len(x))

def funcWithError(x, f=f, percent=1):
    errorPercent = random.uniform(-percent, percent)
    return f(x) * (1 + errorPercent / 100)


def Lagrange(x, X, f):
    L = 0
    for i in range(len(X)):
        l_k = 1
        for j in range(len(X)):
            if j != i:
                l_k *= (x - X[j]) / (X[i] - X[j])
        L += l_k * f(X[i])
    return L


def Legendre(x, X, f, deg=3):
    numOfCoeff = deg + 1
    L = [1, -x, (3 * x**2 - 1) / 2, -(5 * x**3 - 3 * x) / 2]
    CkTop = [
        -0.0850018527987160, 0.4734809926878986, -0.0324755305596799,
        -0.0514968902272397
    ]
    CkBot = [2, 0.6666666666666667, 0.4, 0.2857142857142857]

    result = 0
    for i in range(numOfCoeff):
        result += (CkTop[i]/CkBot[i])*L[i]
    return result