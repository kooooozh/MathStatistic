import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

Data = [[8.432, 6.077, 8.982, 2.897, 5.784, 29.159, 14.564, 9.869, 11.621, 10.405],
        [11.589, 15, 8.659, 15.142, 5.345, 7.69, 16.504, 8.742, 2.834, 9.846],
        [13.697, 13.167, 6.846, 6.727, 12.614, 3.655, 19.627, 1.572, 11.4, 8.384],
        [4.807, 7.897, 13.881, 6.244, 13.188, 12.419, 8.279, 9.37, 7.233, 15.503],
        [13.47, 8.511, 10.134, 18.831, 9.422, 3.715, 8.63, 6.99, 10.187, 17.491],
        [8.262, 11.019, 13.02, 7.619, 5.1, 7.757, 7.772, 16.063, 8.736, 12.214],
        [7.998, 5.651, 5.225, 8.525, 12.935, 9.312, 5.12, 6.146, 19.032, 15.049],
        [9.643, 8.633, 4.367, 3.91, 5.957, 6.345, 14.577, 9.878, 13.658, 4.104],
        [5.752, 5.693, 15.207, 6.776, 5.055, 4.96, 35.182, 14.399, 13.383, 14.144],
        [3.77, 6.394, 11.695, 8.474, 13.196, 7.636, 11.219, 10.463, 6.802, 10.135]]

# Крайние члены вариационного ряда
max_Data = np.max(Data)
min_Data = np.min(Data)

# Размах выборки
omega = max_Data - min_Data

print('Находим крайние члены вариационного ряда.')
print(f'Максимум: {max_Data} \nМинимум: {min_Data}')
print(f'Размах выборки: {omega}')

# Число интервалов группировки по правилу Стерджиса
l = math.trunc(1 + math.log(100, 2))
print(f'Число интервалов: {l}')

# Построение гистограммы относительных частот
h = (max_Data - min_Data) / l
print(f'Ширина интервала: {round(h, 3)}')

print('Границы интервалов:')
intervals = [min_Data for i in range(l+1)]
for i in range(l+1):
    intervals[i] = intervals[i] + i * h
    print(round(intervals[i], 2), end=' - ')

print('\nСередины интервалов:')
middles = [0 for i in range(l)]
for i in range(l):
    middles[i] = intervals[i] + h / 2
    print(round(middles[i], 2), end=' - ')

frequencies = [0 for i in range(l)]

for i in range(l):
    for j in range(10):
        for k in range(10):
            if intervals[i] <= Data[j][k] <= intervals[i+1]:
                frequencies[i] += 1

rel_frequencies = [0 for i in range(l)]

print('\nКоличество попаданий в интервал:')
for i in range(l):
    print(f'{i+1}: {frequencies[i]}')
    rel_frequencies[i] = frequencies[i] / 100
print('Относительные частоты:')
for i in range(l):
    print(f'{i+1}: {rel_frequencies[i]}')

midpoints = [(intervals[i] + intervals[i+1]) / 2 for i in range(len(intervals)-1)]

plt.bar(midpoints, rel_frequencies, width=(intervals[1] - intervals[0]), align='center', label='Гистограмма', edgecolor='black')

plt.plot(midpoints, rel_frequencies, linestyle='-', color='red', label='Полигон')

plt.xlabel("Значения")
plt.ylabel("Относительная частота")
plt.title('Гистограмма относительных частот')
plt.show()

# Выборочное среднее и выборочная дисперсия
Data_mean = np.mean(Data)

Data_var = 0

for i in range(10):
    for j in range(10):
        Data_var += (Data[i][j] - Data_mean) ** 2

Data_var /= 99

print(f'Выборочное среднее: {round(Data_mean, 3)}')
print(f'Выборочная дисперсия: {round(Data_var, 3)}')

''' По виду гистограммы заключаем, что распределение эмпирических частот похоже на гамма-распределение
Оценка параметра
E(X) = lambda / alpha, D(X) = lambda / (alpha)^2 '''

lam = (Data_mean * Data_mean) / Data_var
alpha = Data_mean / Data_var

print('Оценка параметров')
print(f'Лямбда: {round(lam, 3)}')
print(f'Альфа: {round(alpha, 3)}')

# Построение гамма-распределения
Gam = gamma.rvs(a=lam, scale=1/alpha, size=10000)
hist, bin_edges = np.histogram(Gam, bins=intervals, density=True)
hist = hist / np.sum(hist)
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor ='black')
x = np.linspace(min(intervals), max(intervals), 1000)
pdf = gamma.pdf(x, a=lam, scale=1/alpha)
pdf *= h
plt.plot(x, pdf, color = 'red')
plt.xlabel('Значения')
plt.ylabel('Относительная частота')
plt.title('Гистограмма и график гамма-распределения')
plt.show()

# Построение эмпирической функции распределения.
# Строим ступенчатую функцию распределений
Array = []
for i in range(10):
    for j in range(10):
        Array.append(Data[i][j] + round(h / 2, 3))
Array.sort()
plt.hist(Array, histtype='step', cumulative=True, bins=7, density=True, label='Эмпирическая функция распределения')

x = np.linspace(min(intervals), max(intervals), 1000)
cdf = gamma.cdf(x, a=lam, scale=1/alpha)
plt.plot(x, cdf, label='Теоретическая функция распределения')
plt.xlabel('Значение')
plt.ylabel('Относительная частота')
plt.title('Функции распределения')
plt.legend()
plt.show()
