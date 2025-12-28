import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Установка параметров шрифта
plt.rcParams.update({
    'font.size': 12,  # Размер основного шрифта
    'axes.titlesize': 16,  # Размер шрифта заголовка оси
    'axes.labelsize': 14,  # Размер шрифта подписей осей
    'xtick.labelsize': 12,  # Размер шрифта меток по оси X
    'ytick.labelsize': 12,  # Размер шрифта меток по оси Y
    'legend.fontsize': 12,  # Размер шрифта легенды
    'figure.titlesize': 18  # Размер шрифта заголовка figure (если используется)
})

df = pd.read_csv('optimization_results.csv')

# Создаём маппинг для переименования методов и функций
method_mapping = {
    'Base_Sornyak': 'Сорняковый метод',
    'CoordinateDescent': 'Координатный спуск',
    'GradientDescent': 'Градиентный спуск',
    'NelderMead': 'Нелдера-Мида',
    'RandomSearch': 'Случайный поиск',
    'Sornyak_AdaptiveSpread': 'Адаптивный',
    'Sornyak_DynamicPopulation': 'Динамический',
    'Sornyak_Elitism': 'Элитизм',
    'Sornyak_Tournament': 'Турнирный'
}

function_mapping = {
    'Rastrigin': 'Растригина',
    'Rosenbrock': 'Розенброка',
    'Schwefel': 'Швефеля'
}

# Применяем маппинг к DataFrame
df['method'] = df['method'].map(method_mapping)
df['function'] = df['function'].map(function_mapping)

# Создаем сводные таблицы
pivot_iterations = df.pivot_table(
    values='iteration_step',
    index='method',
    columns='function',
    aggfunc='mean'
)

pivot_error = df.pivot_table(
    values='error',
    index='method',
    columns='function',
    aggfunc='mean'
)

# --- График 1: Среднее количество итераций ---
plt.figure(figsize=(10, 8))

sns.heatmap(
    pivot_iterations,
    annot=True,
    fmt='.2f',
    cbar_kws={'label': 'Среднее число итераций'},
    cmap='viridis'
)
plt.title('Среднее число итераций по семенам для каждой функции и метода')
plt.xlabel('Функция')
plt.ylabel('Метод')
plt.tight_layout()
plt.show()

# --- График 2: Средняя ошибка ---
plt.figure(figsize=(10, 8))

sns.heatmap(
    pivot_error,
    annot=True,
    fmt='.2f',
    cbar_kws={'label': 'Средняя ошибка'},
    cmap='viridis'
)
plt.title('Средняя ошибка по семенам для каждой функции и метода')
plt.xlabel('Функция')
plt.ylabel('Метод')
plt.tight_layout()
plt.show()