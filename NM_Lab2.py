from copy import deepcopy

A = [
    [12, -3, -1, 3],
    [-3, 15, 5, -5],
    [-1, 5, 10, 2],
    [3, -5, 2, 11]
]
b = [-26, -55, -58, -24]
n = 4

def solve(A_, b_):
    A = deepcopy(A_)
    b = deepcopy(b_)
    n = len(A)
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

        bb = deepcopy(b)
        for i in range(n):
            b[i] = sum([U[i][j] * bb[j] for j in range(n)])
        
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum([A[i][j] * x[j] for j in range(i + 1, n)])) / A[i][i]
    return x


def inv(B):
    n = len(B)
    X = [[0] * n for _ in range(n)]
    for i in range(n):
        f = [1 if i == j else 0 for j in range(n)]
        x = solve(B, f)
        for j in range(n):
            X[j][i] = x[j]
    return X


def sor(A_, b_, x_, eps, w):
    A = deepcopy(A_)
    b = deepcopy(b_)
    x = deepcopy(x_)
    n = len(b)
    
    B = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            B[i][j] = -A[i][j] / A[i][i]
    c = [0] * n
    for i in range(n):
        c[i] = b[i] / A[i][i]

    while True:
        R = [0] * n
        R_max = 0
        i_max = 0
        for i in range(n):
            R[i] = c[i] - x[i] + sum([B[i][j] * x[j] for j in range(n) if i != j])
            if abs(R[i]) > R_max:
                R_max = abs(R[i])
                i_max = i
    
        x[i_max] += w * R[i_max]

        if abs(w * R[i_max]) < eps:
            break

    return x


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
x = sor(A, b, x, 1e-6, 1.5)
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

A_inv = inv(A)
M = max([sum([abs(A[i][j]) for i in range(n)]) for j in range(n)]) * max([sum([abs(A_inv[i][j]) for i in range(n)]) for j in range(n)])
print("Число обусловленности матрицы:", M)

