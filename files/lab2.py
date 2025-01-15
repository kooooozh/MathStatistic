import math
import random
import matplotlib.pyplot as plt
import numpy as np


def c_j_k(j, k):
    return math.factorial(k) / (math.factorial(j) * math.factorial(k - j))


def check_value(cumul, y):
    X = []
    for item in y:
        tmp = 0
        for prob in cumul:
            if item < prob:
                X.append(tmp)
                break
            tmp += 1
    return X


# Количество испытаний
k = 10
# Вероятность успеха в одном испытании
p = 0.4
# Объем выборки
n = 120

# Находим теоретический закон по формуле Бернулли
prob = []
for j in range(11):
    prob.append(c_j_k(j, k) * (p ** j) * ((1 - p) ** (k - j)))

print('Теоретический закон распределения:')
print('Значение СВ - Вероятность')
for i in range(len(prob)):
    print(f'{i}: {prob[i]}')

# Кумулятивные вероятности
u = [prob[0]]
for i in range(1, 11):
    tmp = u[i - 1] + prob[i]
    u.append(tmp)

print('Кумулятивные вероятности')
u_str = f'u = ('
for item in u:
    if item != 1:
        u_str += f'{round(item, 5)}, '
    else:
        u_str += f'{round(item, 5)})'

print(u_str)

# Моделируем вектор из n случайных чисел
y = []
for i in range(n):
    y.append(random.random())

print('Вектор y:')
print(y)

# По вектору y разыгрываем вектор X:
X = check_value(u, y)
print('Вектор x:')
print(X)

# Построение статистического ряда

count = []
for i in range(11):
    count.append(X.count(i))

freq = [count[i] / n for i in range(11)]
add_freq = [freq[0]]
for i in range(1, 11):
    tmp = add_freq[i - 1] + freq[i]
    add_freq.append(tmp)

print('Значения СВ  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10  |')
print(f'Частоты | {count[0]} | {count[1]} | {count[2]} | {count[3]} | {count[4]} | {count[5]} | {count[6]} | '
      f'{count[7]} | {count[8]} | {count[9]} | {count[10]} |')
print(f'Относительные частоты | {freq[0]} | {freq[1]} | {freq[2]} | {freq[3]} | {freq[4]} | {freq[5]} | {freq[6]} '
      f'| {freq[7]} | {freq[8]} | {freq[9]} | {freq[10]} |')
print(f'Накопленные частоты |{add_freq[0]}|{add_freq[1]}|{add_freq[2]}|{add_freq[3]}|{add_freq[4]}|{add_freq[5]}|'
      f'{add_freq[6]}|{add_freq[7]}|{add_freq[8]}|{add_freq[9]}|{add_freq[10]}|')

# По накопленным частотам строим эмпирическую функцию распределения
x = np.arange(len(add_freq))

# Построение эмпирической функции распределения
plt.step(x, add_freq, label='Эмпирическая функция распределения', where='post')

# Построение биноминального распределения
plt.step(x, u, label='Теоретическая функция распределения', color='red', where='post')

# Настройка графика
plt.title('Эмпирическая и теоретическая функции распределения')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()

# Вычисление статистики Колмогорова
delta = 0
tmp = 0
for i in range(11):
    tmp = abs(u[i] - add_freq[i])
    if tmp > delta:
        delta = tmp

print(f'Статистика Колмогорова: {delta}')

# Выборочные характеристики, сравнение с истинными значениями
x_mean = sum(X) / n
s = sum([(item - x_mean) ** 2 for item in X]) / (n - 1)

print(f'Выборочное среднее: {x_mean}')
print(f'Выборочная дисперсия: {s}')

m = sum([prob[i] * i for i in range(len(prob))])
d = p * (1 - p) * k

print(f'Математическое ожидание: {m}')
print(f'Дисперсия: {d}')
