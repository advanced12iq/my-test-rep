import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def schwefel(x, y):
    return -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

# Создаём директорию plots, если её нет
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('fixed_iterations_results.csv')

# Словарь для хранения русских названий методов
method_names_ru = {
    'Base_Sornyak': 'Базовый Сорняковый метод',
    'CoordinateDescent': 'Координатный спуск',
    'GradientDescent': 'Градиентный спуск',
    'NelderMead': 'Нелдера-Мида',
    'RandomSearch': 'Случайный поиск',
    'Sornyak_AdaptiveSpread': 'Сорняковый метод Адаптивное Распространение',
    'Sornyak_DynamicPopulation': 'Сорняковый метод Динамическая Популяция',
    'Sornyak_Elitism': 'Сорняковый метод Элитизм',
    'Sornyak_Tournament': 'Сорняковый метод Турнир'
}

# Обновляем названия методов в DataFrame
df['method'] = df['method'].map(lambda x: method_names_ru.get(x, x))

functions = df['function'].unique()

function_map = {
    'Rosenbrock': rosenbrock,
    'Rastrigin': rastrigin,
    'Schwefel': schwefel
}

minima = {
    'Rosenbrock': (1, 1),
    'Rastrigin': (0, 0),
    'Schwefel': (420.9687, 420.9687)
}

for func_name in functions:
    func_data = df[df['function'] == func_name]
    
    # Получаем уникальные методы и семена
    all_methods = func_data['method'].unique()
    seeds = func_data['seed'].unique()
    
    # Выбираем первые 3 метода (или все, если их меньше)
    selected_methods = all_methods[:3]
    
    # Параметры сетки: 3 метода на 3 столбца
    n_methods_to_plot = len(selected_methods)
    n_cols_per_row = 3
    n_seeds_to_plot = min(3, len(seeds))  # Берём не более 3 семян
    n_rows = n_methods_to_plot  # Одна строка на метод
    n_cols = n_seeds_to_plot    # Один столбец на семя
    
    # Если методов меньше 3, добавляем пустые subplot'ы для заполнения сетки 3x3
    total_plots_needed = 3 * n_seeds_to_plot
    
    fig, axes = plt.subplots(n_rows, n_seeds_to_plot, figsize=(6 * n_seeds_to_plot, 6 * n_rows))
    
    # Обеспечиваем, чтобы axes был двумерным массивом, даже если n_rows или n_cols = 1
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_seeds_to_plot == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 0 or n_seeds_to_plot == 0:
        # На всякий случай, если данные отсутствуют
        continue

    # Установка диапазона осей в зависимости от функции
    if func_name == 'Rosenbrock':
        x_range = (-2, 2)
        y_range = (-1, 3)
    elif func_name == 'Rastrigin':
        x_range = (-5, 5)
        y_range = (-5, 5)
    else:  # Schwefel
        x_range = (-500, 500)
        y_range = (-500, 500)
    
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = function_map[func_name](X, Y)
    
    # Определяем русское название функции для заголовка
    func_name_ru_map = {'Rosenbrock': 'Розенброк', 'Rastrigin': 'Растригин', 'Schwefel': 'Швефель'}
    func_name_ru = func_name_ru_map.get(func_name, func_name)
    
    for i, method_name in enumerate(selected_methods):
        for j, seed in enumerate(seeds[:n_seeds_to_plot]): # Берём только нужное количество семян
            ax = axes[i, j]
            
            # Рисуем контурную диаграмму функции
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
            
            # Фильтруем данные для конкретного метода и семени
            subset = func_data[(func_data['method'] == method_name) & (func_data['seed'] == seed)]
            if subset.empty:
                print(f"Предупреждение: Нет данных для функции '{func_name_ru}', метода '{method_name}', семени {seed}")
                ax.text(0.5, 0.5, 'Нет данных', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(x_range[0], x_range[1])
                ax.set_ylim(y_range[0], y_range[1])
                continue
            
            subset = subset.sort_values('iteration_step')
            
            x_vals = subset['x1'].values
            y_vals = subset['x2'].values
            iteration_steps = subset['iteration_step'].values
            
            # Уменьшаем количество точек, если их слишком много
            if len(x_vals) > 1000:
                x_vals = x_vals[::10]
                y_vals = y_vals[::10]
                iteration_steps = iteration_steps[::10]
            
            # Рисуем траекторию
            ax.plot(x_vals, y_vals, 'r--', linewidth=0.8, alpha=0.5, zorder=4, label='Траектория')
            
            # Рисуем точки, где цвет зависит от номера итерации
            scatter = ax.scatter(x_vals, y_vals, c=iteration_steps, cmap='plasma', s=30, edgecolors='black', zorder=5, alpha=0.7, label='Итерации')
            
            # Отмечаем глобальный минимум
            min_x, min_y = minima[func_name]
            ax.plot(min_x, min_y, 'g*', markersize=15, label='Глобальный минимум', zorder=10)
            
            # Настройка графика
            ax.set_title(f'Семя {seed}')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.grid(True, alpha=0.3)
            
            # Добавляем название метода слева от первой колонки
            if j == 0:
                ax.text(-0.3, 0.5, method_name, rotation=90, verticalalignment='center', 
                        horizontalalignment='center', transform=ax.transAxes, fontsize=12, weight='bold')
    
    # Добавляем общий заголовок для всей фигуры
    plt.suptitle(f'Оптимизация функции {func_name_ru}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect оставляет место для suptitle
    
    # Сохраняем фигуру
    filename = f'plots/{func_name_ru}_grid.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

print("Все графики сохранены в папку 'plots'.")