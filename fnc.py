import numpy as np
import math
import itertools


def Lagrange(x, matrix, f):
    L = 0
    for i in range(len(matrix)):
        l_k = 1
        for j in range(len(matrix)):
            if j != i:
                l_k *= (x - matrix[j]) / (matrix[i] - matrix[j])
        L += l_k * f(matrix[i])
    return L


def cheb(n, a):
    matrix = []
    for i in range(n):
        matrix.append(a * np.cos((2 * i + 1) / (2 * n) * np.pi))
    return matrix


def E_L(x, X, f):
    maxerror = 0
    for i in x:
        diff = abs(f(i) - Lagrange(i, X, f))
        if diff > maxerror:
            maxerror = diff
    return maxerror


def DividedDifference(X, df, m):
    n = len(X)
    dd = np.zeros((n * m, n * m))
    z = np.zeros(n * m)
    k = 0
    for i in range(n):
        for j in range(m):
            k = i * m + j
            z[k] = X[i]
            dd[k, 0] = df[i, 0]
            for l in range(1, k + 1):
                if (dd[k, l - 1] == dd[k - 1, l - 1]) and (z[k] == z[k - l]):
                    dd[k, l] = df[i, l] / math.factorial(l)
                else:
                    dd[k, l] = (dd[k, l - 1] - dd[k - 1, l - 1]) / (z[k] -
                                                                    z[k - l])
    return dd


def Hermite(x, X, F):
    m = len(F)
    n = len(X)
    matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            matrix[i, j] = F[j](X[i])
    dd = DividedDifference(X, matrix, m)
    P = 0
    u = 1
    for i in range(n):
        for j in range(0, m):
            l = i * m + j
            P += dd[l, l] * u
            u *= (x - X[i])
    return P


def Hermite_derivative(x, X, F):
    m = len(F)
    n = len(X)
    matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            matrix[i, j] = F[j](X[i])
    dd = DividedDifference(X, matrix, m)
    dHermite = 0
    nodesToCombine = np.zeros(n * m)
    for i in range(n):
        for j in range(m):
            k = i * m + j
            nodesToCombine[k] = X[i]
    for i in range(n):
        for j in range(m):
            l = i * m + j + 1
            if l == m * n:
                break
            u = 0
            combinations = itertools.combinations(nodesToCombine[:l].tolist(),
                                                  l - 1)
            for comb in combinations:
                numeratorToSum = 1
                for element in comb:
                    numeratorToSum *= x - element
                u += numeratorToSum
            dHermite += dd[l, l] * u

    return dHermite


def E_Hermite(x, X, F):
    max = 0
    for i in x:
        diff = abs(F[0](i) - Hermite(i, X, F))
        if diff > max:
            max = diff
    return max


def E_Hermite_derivative(x, X, F):
    max = 0
    for i in x:
        H_der = Hermite_derivative(i, X, F)
        diff = abs(F[1](i) - H_der)
        if diff > max:
            max = diff
    return max


def L_derivative(x, X, f):
    L_der = 0
    for i in range(len(X)):
        numerator = 0
        denominator = 1
        for j in range(len(X)):
            numeratorToSum = 1
            if (j != i):
                denominator *= X[i] - X[j]
                for k in range(len(X)):
                    if k != j and k != i:
                        numeratorToSum *= x - X[k]
                numerator += numeratorToSum
        L_der += numerator / denominator * f(X[i])
    return L_der


def E_L_derivative(x, X, F):
    max = 0
    for i in x:
        L_der = L_derivative(i, X, F[0])
        diff = abs(F[1](i) - L_der)
        if diff > max:
            max = diff
    return max


def L_derivative2(x, X, f):
    L_der2 = 0
    for i in range(len(X)):
        numerator = 0
        denominator = 1
        for j in range(len(X)):
            if (j != i):
                denominator *= X[i] - X[j]
        nodesToCombine = X[:i].tolist() + X[i + 1:].tolist()
        combinations = itertools.combinations(nodesToCombine,
                                              len(nodesToCombine) - 2)

        for comb in combinations:
            numeratorToSum = 1
            for element in comb:
                numeratorToSum *= x - element
            numerator += numeratorToSum

        L_der2 += numerator / denominator * f(X[i])

    return L_der2


def Hermite_derivative2(x, X, F):
    m = len(F)
    n = len(X)
    matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            matrix[i, j] = F[j](X[i])
    dd = DividedDifference(X, matrix, m)
    dHermite = 0
    nodesToCombine = np.zeros(n * m)
    for i in range(n):
        for j in range(m):
            k = i * m + j
            nodesToCombine[k] = X[i]
    for i in range(n):
        for j in range(m):
            l = i * m + j + 2
            if l == m * n:
                break
            u = 0
            combinations = itertools.combinations(nodesToCombine[:l].tolist(),
                                                  l - 2)
            for comb in combinations:
                numeratorToSum = 1
                for element in comb:
                    numeratorToSum *= x - element
                u += numeratorToSum
            dHermite += dd[l, l] * u

    return dHermite


def E_Hermite_derivative2(x, X, F):
    max = 0
    for i in x:
        H_der = Hermite_derivative2(i,X, F)
        diff = abs(F[2](i) - H_der)
        if diff > max:
            max = diff
    return max



def E_L_derivative2(x, X, F):
    max = 0
    for i in x:
        L_der = L_derivative2(i, X, F[0])
        diff = abs(F[2](i) - L_der)
        if diff > max:
            max = diff
    return max


def S10(x, X, f):
    for i in range(len(X)):
        if x == X[i]:
            return f(X[i])
        elif (x > X[i]) and (x < X[i + 1]):
            return f(X[i]) * (X[i + 1] - x) / (X[i + 1] - X[i]) + f(
                X[i + 1]) * (x - X[i]) / (X[i + 1] - X[i])


def S31(x, X, f, f_1):
    for i in range(len(X)):
        if x == X[i]:
            return f(X[i])
        elif (x > X[i]) and (x < X[i + 1]):
            h = X[i + 1] - X[i]
            return f(X[i])*(X[i+1]-x)**2*(2*(x-X[i])+h)/h**3 + f(X[i+1])*(x-X[i])**2*(2*(X[i+1]-x)+h)/h**3 \
                     + f_1(X[i])*(X[i+1]-x)**2*(x-X[i])/h**2 + f_1(X[i+1])*(x-X[i])**2*(x-X[i+1])/h**2

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
    nul_lines_U = 0
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


def S32_natural(x, X, f):
    n = len(X)
    F = np.zeros((n, 1))
    for i in range(n):
        F[i] = f(X[i])
    H = np.zeros((n, n))
    h = X[1] - X[0]
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    for i in range(1, n - 1):
        H[i, i - 1] = -3 / h
        H[i, i + 1] = 3 / h
        A[i, i] = 4
        A[i, i + 1] = 1
        A[i, i - 1] = 1
    b = H @ F
    l, u, p, q = LU(A, n)
    M = AxB(l, u, p.dot(b), n)
    M = q @ M
    M[0] = -M[1] / 2 + 3 / 2 * (f(X[1]) - f(X[0])) / h
    M[n - 1] = -M[n - 1] / 2 + 3 / 2 * (f(X[n - 1]) - f(X[n - 2])) / h
    for i in range(n):
        if x == X[i]:
            return f(X[i])
        elif (x >= X[i]) and (x < X[i + 1]):
            h = X[i + 1] - X[i]
            return f(X[i])*(X[i+1]-x)**2*(2*(x-X[i])+h)/h**3 + f(X[i+1])*(x-X[i])**2*(2*(X[i+1]-x)+h)/h**3 \
                     + M[i]*(X[i+1]-x)**2*(x-X[i])/h**2 + M[i+1]*(x-X[i])**2*(x-X[i+1])/h**2


def S32_der(x, X, f, f_1):
    n = len(X)
    F = np.zeros((n, 1))
    for i in range(n):
        F[i] = f(X[i])
    H = np.zeros((n, n))
    h = X[1] - X[0]
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    for i in range(1, n - 1):
        H[i, i - 1] = -3 / h
        H[i, i + 1] = 3 / h
        A[i, i] = 4
        A[i, i + 1] = 1
        A[i, i - 1] = 1
    b = H @ F
    b[0] = f_1(X[0])
    b[n - 1] = f_1(X[n - 1])
    l, u, p, q = LU(A, n)
    M = AxB(l, u, p.dot(b), n)
    M = q @ M
    # построение полинома
    for i in range(n):
        if x == X[i]:
            return f(X[i])
        elif (x >= X[i]) and (x < X[i + 1]):
            h = X[i + 1] - X[i]
            return f(X[i])*(X[i+1]-x)**2*(2*(x-X[i])+h)/h**3 + f(X[i+1])*(x-X[i])**2*(2*(X[i+1]-x)+h)/h**3 \
                     + M[i]*(X[i+1]-x)**2*(x-X[i])/h**2 + M[i+1]*(x-X[i])**2*(x-X[i+1])/h**2