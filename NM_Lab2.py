from copy import deepcopy
import numpy as np

A = [
    [12, -3, -1, 3],
    [-3, 15, 5, -5],
    [-1, 5, 10, 2],
    [3, -5, 2, 11]
]
b = [-26, -55, -58, -24]
n = 4

def solve(A_, b_, f=True):
    A = deepcopy(A_)
    b = deepcopy(b_)
    n = len(A)
    Q = [[0] * n for i in range(n)]
    for i in range(n):
        Q[i][i] = 1
    for l in range(n - 1):
        y = [A[k][l] for k in range(l, n)]
        if all([v == 0 for v in y[1:]]):
            continue
        w = deepcopy(y)
        w[0] = w[0] - sum([v * v for v in y]) ** 0.5
        norm_w = sum([v * v for v in w]) ** 0.5
        w = [v / norm_w for v in w]

        m = n - l
        V = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(m):
                V[i][j] = -2 * w[i] * w[j]
                if i == j:
                    V[i][j] += 1

        U = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j and 0 <= i < l:
                    U[i][j] = 1
                elif l <= i and l <= j:
                    U[i][j] = V[i - l][j - l]

        AA = deepcopy(A)
        for i in range(n):
            for j in range(n):
                A[i][j] = sum([U[i][t] * AA[t][j] for t in range(n)])

        QQ = deepcopy(Q)
        for i in range(n):
            for j in range(n):
                Q[i][j] = sum([U[i][t] * QQ[t][j] for t in range(n)])

        bb = deepcopy(b)
        for i in range(n):
            b[i] = sum([U[i][j] * bb[j] for j in range(n)])

    if f:
        print("Ортогональная")
        for line in Q:
            for el in line:
                print(f"{' ' if el > 0 else ''}{el:.1f}", end=' ')
            print()
        print()

        print("Верхнетреугольная")
        for line in A:
            for el in line:
                print(f"{' ' if el > 0 else ''}{el:.1f}", end=' ')
            print()
        print()
        
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum([A[i][j] * x[j] for j in range(i + 1, n)])) / A[i][i]
    return x


def inv(B):
    n = len(B)
    X = [[0] * n for _ in range(n)]
    for i in range(n):
        f = [1 if i == j else 0 for j in range(n)]
        x = solve(B, f, False)
        for j in range(n):
            X[j][i] = x[j]
    return X


def sor(A_, b_, x_, eps, w):
    A = deepcopy(A_)
    f = deepcopy(b_)
    x = deepcopy(x_)
    n = len(b)

    cnt = 0

    while True:
        cnt += 1
        xx = deepcopy(x)

        for i in range(n):
            x[i] = -w * sum([A[i][j] / A[i][i] * x[j] for j in range(i)]) + (1 - w) * xx[i] - \
                w * sum([A[i][j] / A[i][i] * xx[j] for j in range(i + 1, n)]) + w * f[i] / A[i][i]

        if sum([abs(x[i] - xx[i]) for i in range(n)]) < eps:
            break

    return x, cnt


x = solve(A, b)
print("Решение СЛАУ:")
for v in x:
    print(f'{v:.10f}')
print()

r = [0] * n
for i in range(n):
    r[i] = sum([A[i][j] * x[j] for j in range(n)]) - b[i]
print("Вектор невязки:")
for v in r:
    print(f'{v:.30f}')
print()

x = [int(v) - 1 if v < 0 else int(v) for v in x]
x, cnt = sor(A, b, x, 1e-6, 1.2)
print("Решение СЛАУ:")
for v in x:
    print(f'{v:.10f}')
print("Число шагов:", cnt)
print()

r = [0] * n
for i in range(n):
    r[i] = sum([A[i][j] * x[j] for j in range(n)]) - b[i]
print("Вектор невязки:")
for v in r:
    print(f'{v:.30f}')
print()

A_inv = inv(A)
M = max([sum([abs(A[i][j]) for i in range(n)]) for j in range(n)]) * max([sum([abs(A_inv[i][j]) for i in range(n)]) for j in range(n)])
print("Число обусловленности матрицы:", M, np.linalg.cond(A, p=1))

