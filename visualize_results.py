import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('optimization_results.csv')

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    pivot_iterations,
    annot=True,
    fmt='.2f',
    ax=ax1,
    cbar_kws={'label': 'Mean Iteration Steps'},
    cmap='viridis'
)
ax1.set_title('Mean Iteration Steps Across Seeds for Each Function and Method')

sns.heatmap(
    pivot_error,
    annot=True,
    fmt='.2f',
    ax=ax2,
    cbar_kws={'label': 'Mean Error'},
    cmap='viridis'
)
ax2.set_title('Mean Error Across Seeds for Each Function and Method')

plt.tight_layout()
plt.show()