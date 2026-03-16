import numpy as np
from copy import deepcopy

A = [
    [0.34, 1.17, 0.2, 8.13],
    [3.52, 4.73, 4.37, 5.89],
    [-6.25, 2.54, 6.91, -5.43],
    [-2.13, 2.21, 4.17, 6.11]
]
f = [4.15, 2.92, -3.14, 7.65]


def solve(B, t):
    A = deepcopy(B)
    f = deepcopy(t)
    n = len(A)
    for i in range(n):
        k = i
        for j in range(i, n):
            if A[j][i] != 0:
                k = j
                break
        A[k], A[i] = A[i], A[k]
        f[k], f[i] = f[i], f[k]
        v = A[i][i]
        for j in range(i, n):
            A[i][j] /= v
        f[i] /= v
        for j in range(i + 1, n):
            u = A[j][i]
            f[j] -= u * f[i]
            for s in range(i, n):
                A[j][s] -= u* A[i][s]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = f[i] - sum([A[i][j] * x[j] for j in range(i + 1, n)])
    return x


def det(B):
    A = deepcopy(B)
    n = len(A)
    for i in range(n):
        k = i
        for j in range(i, n):
            if A[j][i] != 0:
                k = j
                break
        A[k], A[i] = A[i], A[k]
        for j in range(i + 1, n):
            u = A[j][i]
            for s in range(i, n):
                A[j][s] -= u * A[i][s] / A[i][i]
    d = 1
    for i in range(n):
        d *= A[i][i]
    return d


def inv(B):
    n = len(B)
    X = [[0] * n for _ in range(n)]
    for i in range(n):
        f = [1 if i == j else 0 for j in range(n)]
        x = solve(B, f)
        for j in range(n):
            X[j][i] = x[j]
    return X
    


x = solve(A, f)
print("Решение СЛАУ:")
for v in x:
    print(f'{v:.4f}')
print()

AA = np.array(A)
ff = np.array(f)
xx = np.array(x)
r = AA @ xx - ff
print("Вектор невязки:")
for v in r:
    print(f'{v:.16f}')
print()

d = det(A)
print(f'Определитель: {d:.4f}, {np.linalg.det(AA):.4f}')
print()

X = inv(A)
print("Обратная матрица:")
for line in X:
    for value in line:
        print(f"{' ' if value >= 0 else ''}{value:.4f}", end=' ')
    print()

print()
print("Проверка. AX=")
E = np.array(X) @ AA
for line in E:
    for value in line:
        print(f"{' ' if value >= 0 else ''}{value:.4f}", end=' ')
    print()
