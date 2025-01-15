import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import root_scalar
from scipy.stats import beta
from scipy.stats import norm


def func_alpha(p):
    return binom.ppf(alpha / 2, n * k, p) - K


def func_1_minus_alpha(p):
    return binom.ppf(1 - alpha / 2, n * k, p) - K


# Выборка
X = [5, 3, 3, 6, 3, 2, 3, 3, 4, 6, 3, 3, 4, 2, 5, 5, 7, 5, 1, 3, 4, 3, 5, 2, 3, 4, 4, 5, 1, 5,
     3, 4, 3, 6, 2, 1, 2, 6, 2, 5, 1, 2, 6, 5, 2, 4, 5, 1, 4, 6, 7, 3, 5, 3, 4, 3, 3, 3, 2, 7,
     4, 6, 4, 5, 3, 5, 2, 3, 3, 2, 3, 5, 3, 6, 3, 5, 7, 1, 4, 3, 2, 6, 5, 4, 4, 4, 4, 8, 4, 4, 3,
     3, 4, 6, 3, 5, 4, 3, 4, 2, 6, 3, 4, 5, 3, 3, 7, 5, 4, 2, 5, 4, 5, 1, 5, 6, 4, 6, 3, 4]

k = 10
n = len(X)
print(n)

# Выборочное среднее
x_mean = np.mean(X)
print(f'Выборочное среднее: {x_mean}')

# Статистика К - суммарное число успехов
K = np.sum(X)
print(f'Суммарное число успехов: {K}')
print(f'Проверка: {x_mean * n}')

# Генерация квантилей
alpha = 0.01
p = np.linspace(0, 1, 1000)
quantiles_alpha = binom.ppf(alpha, n * k, p)
quantiles_1_minus_alpha = binom.ppf(1 - alpha, n * k, p)

# Построение графика
plt.figure(figsize=(8, 6))
plt.plot(p, quantiles_alpha, label=r'$u(p, \alpha)$')
plt.plot(p, quantiles_1_minus_alpha, label=r'$u(p, 1-\alpha)$')
plt.axhline(K, color='black', linestyle='--')
plt.text(0.05, K + 20, f'K={K}', fontsize=12, verticalalignment='center')
plt.title(r'Квантили биноминального распределения для $\alpha=0.01$', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Численное нахождение точек пересечения квантилей с К=466
result_alpha = root_scalar(func_alpha, bracket=[0, 1], method='bisect')
result_1_minus_alpha = root_scalar(func_1_minus_alpha, bracket=[0, 1], method='bisect')

p_alpha = result_alpha.root
p_1_minus_alpha = result_1_minus_alpha.root

print(f'Границы доверительного интервала для К=466: {p_1_minus_alpha}, {p_alpha}')
# Вывод - ОК, т.к. истинное значение p=0.4 опадает в интервал

# Решение уравнений Клоппера-Пирсона для разных уровней значимости
alphas = [0.1, 0.05, 0.02]
for a in alphas:
     lower_bound = beta.ppf(a / 2, K, n * k - K + 1)
     upper_bound = beta.ppf(1 - a / 2, K + 1, n * k - K)
     print(f'Решение уравнений Клоппера-Пирсона, alpha={a}: {lower_bound}, {upper_bound}')

# Решение с использованием ЦПТ
for a in alphas:
     sigma = np.sqrt(x_mean * (k - x_mean) / (n * k)) / k
     z_alpha = norm.ppf(a / 2)
     lower_bound_clt = x_mean / k + z_alpha * sigma
     upper_bound_clt = x_mean / k - z_alpha * sigma
     print(f'Границы интервалов по ЦПТ, alpha={a}: {lower_bound_clt}, {upper_bound_clt}')


# Функции распределения B(k, p), B(k, p1), B(k, p2)
p = 0.4
p1 = 0.365
p2 = 0.412
x_values = np.linspace(0, k, 1000)

binom_cdf_p = binom.cdf(x_values, k, p)
binom_cdf_p1 = binom.cdf(x_values, k, p1)
binom_cdf_p2 = binom.cdf(x_values, k, p2)

plt.plot(x_values, binom_cdf_p, label=r'$B(k=10, p=0.4)$')
plt.plot(x_values, binom_cdf_p1, label=r'$B(k=10, p=0.365)$')
plt.plot(x_values, binom_cdf_p2, label=r'$B(k=10, p=0.412)$')
plt.title(r'Функции распределения $B(k, p)$ для $p=0.4$, $p_1=0.365$, и $p_2=0.412$', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()