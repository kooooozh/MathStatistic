import random
import numpy as np
import matplotlib.pyplot as plt
import math


def reverse_pareto(y: float) -> float:
    return 50 / (1 - y) ** (1 / 20)


def ind(z: float) -> bool:
    if z > 0:
        return 1
    else:
        return 0


def f_emp(z: float, n: int, X: list[float]) -> float:
    res = 0
    for i in range(n):
        res += ind(z - X[i])

    return res


# Константы варианта, дано распределение Парето
n = 120  # объем выборки
# Параметры распределения
k = 20
xm = 50

# Моделируем числа
X = []
for i in range(n):
    X.append(round(random.random(), 6))

print(X)

# Пересчитываем с помощью обратной функции
Y = []
for i in range(len(X)):
    Y.append(round(reverse_pareto(X[i]), 6))

print(Y)

# Обработка данных
Y_exp = k * xm / (k - 1)
D = ((xm / (k - 1)) ** 2) * k / (k - 2)
print(f'Математическое ожидание: {round(Y_exp, 6)}')
print(f'Дисперсия: {round(D, 6)}')

Y_avg = sum(Y) / n
S2 = sum([(item - Y_avg) ** 2 for item in Y]) / (n - 1)
print(f'Выборочное среднее: {Y_avg}')
print(f'Выборочная дисперсия: {S2}')
print(f'Сравнение: {round(abs(Y_avg - Y_exp), 6)}')
print(f'Сравнение: {round((D / S2) ** (1 / 2), 6)}')

delta = (max(Y) - 50) / 7
rel_freq = [0, 0, 0, 0, 0, 0, 0]
for item in Y:
    i = 0
    marg = delta
    while marg < max(Y):
        if item <= 50 + marg:
            rel_freq[i] += 1
            break
        else:
            marg += delta
            i += 1

print(max(Y))
print(rel_freq)

freq = [rel_freq[i] / 120 for i in range(len(rel_freq))]
print(freq)

intervals = np.linspace(50, max(Y), 8)

#plt.bar(intervals[:-1], np.array(rel_freq) / n, width=intervals[1] - intervals[0], align='edge', edgecolor='black')
plt.hist(Y, bins=7, density=True, edgecolor='black')
# Генерация данных
x = np.linspace(xm, max(Y), 1000)  # Диапазон значений x
pdf = (k * (xm ** k)) / (x ** (k + 1))  # Плотность распределения

# Построение графика
plt.plot(x, pdf, color='red')
plt.xlabel('Значения')
plt.ylabel('Относительная частота')
plt.title('Гистограмма относительных частот и теоретическая плотность распределения')
plt.show()

# Построение эмпирической ф-ии распределения и нер-во Дворецкий-Кифер-Волфовиц
gamma = 0.1
epsilon = math.sqrt(-1 / (2 * n) * math.log((gamma / 2), math.e))
print(f'Эпсилон: {epsilon}')

x = np.linspace(50, max(Y), 300)
f = 1 - (xm / x) ** k

z_values = np.linspace(min(Y), max(Y), 100)

# Вычисление значений эмпирической функции распределения
f_emp_values = [f_emp(z, n, Y) / n for z in z_values]
f_emp_values_pe = [f_emp(z, n, Y) / n + epsilon for z in z_values]
f_emp_values_me = [f_emp(z, n, Y) / n - epsilon for z in z_values]

f_emp_values = np.clip(f_emp_values, 0, 1)
f_emp_values_pe = np.clip(f_emp_values_pe, 0, 1)
f_emp_values_me = np.clip(f_emp_values_me, 0, 1)

# Построение ступенчатого графика
plt.step(z_values, f_emp_values, where='post')
plt.step(z_values, f_emp_values_pe, where='post')
plt.step(z_values, f_emp_values_me, where='post')
plt.plot(x, f)
plt.show()
