import numpy as np
import math
import random


def f(x):
    return (1/2 * x.transpose() @ matrix_A) @ x + x.transpose() @ vector_B


def Seidel(matrix_A, matrix_b, eps=1e-5):
    n = len(matrix_A)
    matrix_x = np.zeros((n, 1))
    n_iter = 0
    convergence = False
    while not convergence:
        n_iter += 1
        matrix_x_new = np.copy(matrix_x)

        for i in range(n):
            s1 = sum(matrix_A[i][j] * matrix_x_new[j] for j in range(i))
            s2 = sum(matrix_A[i][j] * matrix_x[j] for j in range(i + 1, n))
            matrix_x_new[i] = (matrix_b[i] - s1 - s2) / matrix_A[i][i]

        convergence = math.sqrt(sum((matrix_x_new[i] - matrix_x[i]) ** 2 for i in range(n))) < eps
        matrix_x = matrix_x_new

        if n_iter > 1000:
            return np.zeros(n), -1

    return matrix_x, n_iter


def gradient_d(X):
    vector_x = np.copy(X)
    count = 0
    while True:
        count += 1

        gradient = matrix_A.dot(vector_x) + vector_B

        if (np.linalg.norm(gradient) == 0):
            print(count)
            break

        step = -np.linalg.norm(gradient) ** 2 / (gradient.transpose().dot(matrix_A)).dot(gradient)
        vector_x += step * gradient

        if (np.linalg.norm(step * gradient) < epsilon):
            print(count)
            break

    return vector_x

def coord_d(X):
    vector_x = np.copy(X)
    old_x = np.copy(X)
    count = 0

    while True:

        for i in range(N):
            step = -(matrix_A[i].dot(vector_x) + vector_B[i]) / matrix_A[i][i]
            vector_x[i] += step
        count += N

        if (np.linalg.norm(vector_x - old_x) < epsilon):
            print(count)
            break

        old_x = np.copy(vector_x)

    return vector_x


N = 3
epsilon = 1e-5


matrix_A1 = np.array([[random.randrange(-2, 2) for y in range(N)] for x in range(N)], dtype=float)

matrix_A = matrix_A1 @ matrix_A1.transpose()

print("Initial matrix")
print(matrix_A)

vector_B = np.array([[random.randrange(-1, 2)] for x in range(N)], dtype=float)
print("Initial vector b")
print(vector_B)

start_x = np.array([[random.randrange(-15, 15)] for x in range(N)], dtype=float)
print("Начальный x")
print(start_x)

print("Seidel:")
print(Seidel(matrix_A, -vector_B, epsilon))

print("Метод наискорейшего градиентного спуска:")
vector_x = gradient_d(start_x)
print(vector_x)


print("Метод наискорейшего покоординатного спуска:")
vector_x = coord_d(start_x)
print(vector_x)
