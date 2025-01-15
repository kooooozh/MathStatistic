import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2, norm

Data = [-21.057,-14.597,-11.356,-13.153,-9.221,-14.814,-10.336,-13.539,-15.996,-15.144,
        -5.179,	-15.539,-19.106,-11.762,-11.226,-14.907,-12.371,-16.74,-9.65,-11.85,
        -8.535,-6.885,-11.704,-12.286,-8.729,-9.687,-3.822,-11.15,-13.873,-9.015,
        -12.949,-10.62,-19.12,-13.785,-11.333,-9.046,-12.113,-10.035,-12.317,-10.029,
        -11.247,-10.116,-11.453,-15.089,-12.566,-11.871,-9.962,-14.874,-12.061,-10.429,
        -15.141,-15.219,-10.488,-12.556,-6.501,-15.068,-9.235,-14.619,-9.286,-15.039,
        -10.143,-13.752,-10.578,-10.234,-15.48,-9.082,-11.872,-12.558,-7.085,-11.626,
        -12.213,-8.93,-12.331,-5.643,-10.078,-14.568,-9.024,-11.993,-17.757,-11.451]

a0 = -11.6
alpha = 0.05
s0 = 3.3
s1 = 3
a1 = -12.6
n = len(Data)
min = min(Data)
max = max(Data)
omega = max - min
print(f'Крайние члены вариационного ряда: минимум - {min},  максимум - {max}')
print(f'Размах выборки: {omega}')

l = 7 # Число интервалов
h = omega / l
print(f'Интервальный шаг: {h}')

borders = [0 for i in range(l+1)]
print('Границы интервалов:')
for i in range(l+1):
    if i == 0:
        borders[i] = min
        print(f'{i}: {borders[i]}')
    else:
        borders[i] = borders[i-1] + h
        print(f'{i}: {borders[i]}')

nu = [0 for i in range(l)]
for elem in Data:
    i = 0
    while elem >= borders[i]:
        i += 1
    nu[i-1] += 1

print('Количество элементов в каждом из интервалов:', nu)
p = [nu[i] / n for i in range(len(nu))]
print('Относительные частоты:', p)
pr = [p[i] / h for i in range(len(p))]
print('Вектор плотности относительной частоты:', pr)

# Гистограмма относительных частот
widths = np.diff(borders)
plt.figure(figsize=(10, 6))
plt.bar(borders[:-1], pr, width=widths, align='edge', edgecolor='black', color='orange')
plt.xlabel('Интервалы (int)')
plt.ylabel('Относительные частоты (pr)')
plt.title('Гистограмма относительных частот')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim([-21.057, -3.822])
plt.show()

avg = sum(Data) / n
print(f'Выборочное среднее: {avg}')

s2 = 0
for elem in Data:
    s2 += (elem - avg) * (elem - avg)

s2 /= (n - 1)
s = s2 ** (1 / 2)
print(f'Выборочная дисперсия: {s2}')
print(f'СКО: {s}')

# №1
C2 = a0 + t.ppf(alpha, df=n-1) * s / (n ** (1/2))
print(C2)

# №2
C3 = s0 * s0 * chi2.ppf(alpha, df=n-1) / (n - 1)
print(C3)

# №3
u005 = -1.64485 # квантиль нормального распределения
C1 = a0 + s1 * u005 / (n ** (1 / 2))
print(C1)

# №4 - посчитать в номотехе квантиль
beta = 1 - 0.90932
print(beta)

# №5
a1b = C1 - 1.644853627 * s1 / (n ** (1/2))
print(a1b)

# №6
x = np.linspace(borders[0], borders[-1], 500)
density_a0 = norm.pdf(x, loc=a0, scale=s1)
density_a1 = norm.pdf(x, loc=a1, scale=s1)
widths = np.diff(borders)
plt.figure(figsize=(10, 6))
plt.bar(borders[:-1], pr, width=widths, align='edge', edgecolor='black', color='orange', alpha=0.7, label='Гистограмма относительных частот')
plt.plot(x, density_a0, color='blue', label=f'N({a0}, {s1})')
plt.plot(x, density_a1, color='green', label=f'N({a1}, {s1})')
plt.xlabel('Интервалы (int)')
plt.ylabel('Относительные частоты (pr)')
plt.title('Гистограмма относительных частот с графиками плотностей нормального распределения')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim([-21.057, -3.822])
plt.legend()
plt.show()
