import math
from NM_Lab1 import solve
from copy import deepcopy

X = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

Y = [
    0.5913,
    0.63 + 4/17,
    0.7162,
    0.8731,
    0.9574,
    1.8 - math.cos(4/11),
    1.3561,
    1.2738,
    1.1 + 4/29,
    1.1672
]

print("X\tY")
print("-" * 14)

for i in range(len(X)):
    print(f"{X[i]}\t{Y[i]:.4f}")

xmin = X[0]
xmax = X[-1]
h = X[1] - X[0]
N = len(X) - 1

def P(t, k, N):
    if k == 0:
        return 1.0
    if k == 1:
        return 1 - 2 * t / N
    if k == 2:
        return 1 - 6 * t / N + 6 * t * (t - 1) / (N * (N - 1))
    if k == 3:
        return 1 - 12 * t / N + \
                30 * t * (t - 1) / (N * (N - 1)) - \
                20 * t * (t - 1) * (t - 2) / (N * (N - 1) * (N - 2))


T = [(x - X[0]) / h for x in X]
m = 4

A = [[0] * m for _ in range(m)]
b = [0] * m

#МНК
for i in range(m):
    for j in range(m):
        for k in range(len(X)):
            A[i][j] += P(T[k], i, N) * P(T[k], j, N)
    for k in range(len(X)):
        b[i] += Y[k] * P(T[k], i, N)
c = solve(A, b)


def approx(x):
    t = (x - X[0]) / h
    return sum(c[i] * P(t, i, N) for i in range(m))

print("\nРазложение через полиномы Чебышева:")
print(f"f(x) = {c[0]:.4f}*P0 + {c[1]:.4f}*P1 + {c[2]:.4f}*P2 + {c[3]:.4f}*P3")
print()

"""
alpha = 2 / (xmax - xmin)
beta = -(xmin + xmax) / (xmax - xmin)

A0 = c[0] - c[2]
A1 = c[1] - 3*c[3]
A2 = 2*c[2]
A3 = 4*c[3]

#d1 = A3 * alpha**3
#d2 = 3*A3*alpha**2*beta + A2*alpha**2
#d3 = 3*A3*alpha*beta**2 + 2*A2*alpha*beta + A1*alpha
#d4 = A3*beta**3 + A2*beta**2 + A1*beta + A0

#print("Разложение через полином 3 степени:")
#print(f"f(x) = {d1:.4f}*x^3 + {d2:.4f}*x^2 + {d3:.4f}*x + {d4:.4f}")
"""

print()
for i in range(len(X)-1):
    xm = X[i] + h/2
    print(f"f({xm:.2f}) = {approx(xm):.4f}")

#------------------------------------------------------------------------------

def max_without_diag(A):
    n = len(A)
    res = 0
    p, q = 0, 1
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j]) > res:
                res = abs(A[i][j])
                p, q = i, j
    return p, q

def jacobi(A, eps=0.001):
    A = deepcopy(A)
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    count = 0
    prev = float('inf')

    while True:
        p, q = max_without_diag(A)
        cur = abs(A[p][q])

        if cur < eps:
            break

        prev = cur

        phi = 0.5 * math.atan2(2 * A[p][q], A[p][p] - A[q][q])
        c = math.cos(phi)
        s = math.sin(phi)

        for i in range(n):
            api = A[i][p]
            aqi = A[i][q]
            A[i][p] = c*api - s*aqi
            A[i][q] = s*api + c*aqi

        for j in range(n):
            apj = A[p][j]
            aqj = A[q][j]
            A[p][j] = c*apj - s*aqj
            A[q][j] = s*apj + c*aqj

        for i in range(n):
            vip = V[i][p]
            viq = V[i][q]
            V[i][p] = c*vip - s*viq
            V[i][q] = s*vip + c*viq

        count += 1

    eigen_values = [A[i][i] for i in range(n)]
    return eigen_values, V, count

A = [
    [1, 1.5, 2.5, 3.5],
    [1.5, 1, 2, 1.6],
    [2.5, 2, 1, 1.7],
    [3.5, 1.6, 1.7, 1]
]

eigen_values, eigen_vecs, count = jacobi(A)

print("Собственные значения")
for i, v in enumerate(eigen_values):
    print(f"λ{i+1} = {v:.4f}")

print()
print("Собственные векторы")
for i in range(len(eigen_values)):
    print(f"\nВектор для λ{i+1} = {eigen_values[i]:.4f}:")
    for j in range(len(eigen_vecs)):
        print(f"{eigen_vecs[j][i]:.4f}")

print(f"\nКоличество итераций Якоби: {count}")

def dot(A, x):
    return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]

def power_method(A, eps=1e-3):
    x = [1] * len(A)
    lm = 0
    count = 0

    while True:
        lm_prev = lm
        y = dot(A, x)
        lm = max(abs(v) for v in y)
        
        x = [v / lm for v in y]

        count += 1

        if abs(lm - lm_prev) < eps:
            break

    return lm, count

radius, count = power_method(A)

print("\nСпектральный радиус")
print(f"{radius:.4f}")
print("Количество итераций степенного метода:", count)
