import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

df = pd.read_csv('optimization_results.csv')
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
df['method'] = df['method'].map(method_mapping)
df['function'] = df['function'].map(function_mapping)
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
plt.savefig('./plots/plot1.png')
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
plt.savefig('./plots/plot2.png')