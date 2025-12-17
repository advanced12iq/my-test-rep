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

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('fixed_iterations_results.csv')

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
    
    methods = func_data['method'].unique()
    seeds = func_data['seed'].unique()
    
    n_methods = len(methods)
    n_seeds = len(seeds)
    n_rows = n_methods
    n_cols = n_seeds
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    if func_name == 'Rosenbrock':
        x_range = (-2, 2)
        y_range = (-1, 3)
    elif func_name == 'Rastrigin':
        x_range = (-5, 5)
        y_range = (-5, 5)
    else:
        x_range = (-500, 500)
        y_range = (-500, 500)
    
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = function_map[func_name](X, Y)
    
    for i, method_name in enumerate(methods):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
            
            subset = func_data[(func_data['method'] == method_name) & (func_data['seed'] == seed)]
            subset = subset.sort_values('iteration_step')
            
            x_vals = subset['x1'].values
            y_vals = subset['x2'].values
            iteration_steps = subset['iteration_step'].values
            
            if len(x_vals) > 1000:
                x_vals = x_vals[::10]
                y_vals = y_vals[::10]
                iteration_steps = iteration_steps[::10]
            
            scatter = ax.scatter(x_vals, y_vals, c=iteration_steps, cmap='plasma', s=30, edgecolors='black', zorder=5, alpha=0.7)
            
            ax.plot(x_vals, y_vals, 'r--', linewidth=0.8, alpha=0.5, zorder=4)
        
            min_x, min_y = minima[func_name]
            ax.plot(min_x, min_y, 'r*', markersize=15, label='Global Minimum', zorder=10)
            
            ax.set_title(f'Seed {seed}')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.grid(True, alpha=0.3)
            
            if j == 0:
                ax.text(-0.3, 0.5, method_name, rotation=90, verticalalignment='center', 
                        horizontalalignment='center', transform=ax.transAxes, fontsize=12, weight='bold')
    
    plt.suptitle(f'{func_name}')
    plt.tight_layout()
    
    filename = f'plots/{func_name}_grid.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()