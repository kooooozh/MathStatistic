import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, norm
from math import exp, pow, log

Data = [-21.057,
-5.179,
-8.535,
-12.949,
-11.247,
-15.141,
-10.143,
-12.213,
-14.597,
-15.539,
-6.885,
-10.62,
-10.116,
-15.219,
-13.752,
-8.93,
-11.356,
-19.106,
-11.704,
-19.12,
-11.453,
-10.488,
-10.578,
-12.331,
-13.153,
-11.762,
-12.286,
-13.785,
-15.089,
-12.556,
-10.234,
-5.643,
-9.221,
-11.226,
-8.729,
-11.333,
-12.566,
-6.501,
-15.48,
-10.078,
-14.814,
-14.907,
-9.687,
-9.046,
-11.871,
-15.068,
-9.082,
-14.568,
-10.336,
-12.371,
-3.822,
-12.113,
-9.962,
-9.235,
-11.872,
-9.024,
-13.539,
-16.74,
-11.15,
-10.035,
-14.874,
-14.619,
-12.558,
-11.993,
-15.996,
-9.65,
-13.873,
-12.317,
-12.061,
-9.286,
-7.085,
-17.757,
-15.144,
-11.85,
-9.015,
-10.029,
-10.429,
-15.039,
-11.626,
-11.451]
print(Data)

a0 = -11.6
alpha = 0.05
sigma_0 = 3.3
sigma_1 = 3
a1 = -12.6
n = len(Data)
beta = 0.09068 # из 5 лабы
C1 = -12.1517 # из 5 лабы

# №1
A = (1 - beta) / alpha
B = beta / (1 - alpha)
print(f'A: {A}, B: {B}')
# №2
def Z(j, data, a0, a1, s1):
    res = 1
    for i in range(j):
        first = (a0 * a0 - a1 * a1) / (2 * s1 * s1)
        second = (a1 - a0) * data[i] / (s1 * s1)
        exp_term = exp(first + second)
        res *= exp_term
    return res

j_values = range(1, len(Data) + 1)
z_values = [Z(j, Data, a0, a1, sigma_1) for j in j_values]

plt.figure(figsize=(10, 6))
plt.plot(j_values, z_values, label="Z(j)", color="blue")
plt.axhline(y=A, color='red', linestyle='--', label=f"A = {A}")
plt.axhline(y=B, color='green', linestyle='--', label=f"B = {B}")
plt.xlabel("j")
plt.ylabel("Z(j)")
plt.legend()
plt.grid(True)
plt.show()

# №3
M0 = -1 * (a1 - a0) ** 2 / (2 * sigma_1 ** 2)
Ma0_nu = (alpha * log(A) + (1 - alpha) * log(B)) / M0
print(f'M0: {M0}')
print(f'Ma0_nu: {Ma0_nu}')
M1 = -M0
Ma1_nu = (beta * log(B) + (1 - beta) * log(A)) / M1
print(f'Ma1_nu: {Ma1_nu}')

# №4

k1 = (sigma_1 ** 2) / (n * (a1 - a0))
k2 = (a1 + a0) / 2
print(k1, k2)
C = exp((C1 - k2) / k1)
print(C)

plt.figure(figsize=(10, 6))
plt.plot(j_values, z_values, label="Z(j)", color="blue")
plt.axhline(y=A, color='red', linestyle='--', label=f"A = {A}")
plt.axhline(y=B, color='green', linestyle='--', label=f"B = {B}")
plt.axhline(y=C, color='orange', linestyle='--', label=f"C = {C}")
plt.xlabel("j")
plt.ylabel("Z(j)")
plt.legend()
plt.grid(True)
plt.show()

print(Z(n, Data, a0, a1, sigma_1))