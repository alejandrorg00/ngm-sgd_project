# -*- coding: utf-8 -*-
"""
Gain tasks plots
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rc('font', family='sans-serif', serif=['Arial'])
plt.rc('axes', edgecolor='black', linewidth=3)
plt.rc('xtick', color='black')
plt.rc('ytick', color='black')
plt.rc('grid', color='black')

# --- Axis limits ---
xlim_class   = (0.9, 3.1)
ylim_class   = (1.2, 2.5)
xlim_domain  = (0.9, 3.1)
ylim_domain  = (2.0, 4.0)

# --- Data configuration ---
datasets = {
    'splitMNIST':      {'file': 'dataAvg_splitMNIST.pkl',     'tasks': 5, 'iters_per_task': 200, 'type': 'class-incremental'},
    'splitCIFAR10':    {'file': 'dataAvg_splitCIFAR10.pkl',   'tasks': 5, 'iters_per_task': 400, 'type': 'class-incremental'},
    'permMNIST':       {'file': 'dataAvg_permMNIST.pkl',      'tasks': 3, 'iters_per_task': 200, 'type': 'domain-incremental'},
    'rotMNIST':        {'file': 'dataAvg_rotMNIST.pkl',       'tasks': 3, 'iters_per_task': 400, 'type': 'domain-incremental'},
    'domainCIFAR100':  {'file': 'dataAvg_domainCIFAR100.pkl', 'tasks': 3, 'iters_per_task': 400, 'type': 'domain-incremental'},
}

method = 'ENTROPY GAIN'
colors = {
    'splitMNIST':     'indianred',
    'splitCIFAR10':   'chocolate',
    'permMNIST':      'goldenrod',
    'rotMNIST':       'orangered',
    'domainCIFAR100': 'lightsalmon',
}
marker = 'o'
fits = {}

# Display names for the legend
display_names = {
    'splitMNIST':     'Split MNIST',
    'splitCIFAR10':   'Split CIFAR-10',
    'permMNIST':      'Permuted MNIST',
    'rotMNIST':       'Rotated MNIST',
    'domainCIFAR100': 'Domain CIFAR-100'
}

# --- fits ---
for name, cfg in datasets.items():
    with open(cfg['file'], 'rb') as f:
        data = pickle.load(f)
    gains = np.array(data[0][method]['gain'])
    mts, its = cfg['tasks'], cfg['iters_per_task']
    arr = gains.reshape(mts, its)
    if cfg['type'] == 'class-incremental':
        arr = arr[:3, :]
        n_tasks = 3
    else:
        n_tasks = mts

    mean_per_task = arr.mean(axis=1)
    std_per_task  = arr.std(axis=1)
    x = np.arange(1, n_tasks + 1)
    m, b = np.polyfit(x, mean_per_task, 1)
    fits[name] = (m, b, mean_per_task, std_per_task)

# --- Plot ---
fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for name, cfg in datasets.items():
    m, b, mean_per_task, std_per_task = fits[name]
    n_tasks = 3 if cfg['type']=='class-incremental' else cfg['tasks']
    x = np.arange(1, n_tasks + 1)
    ax = ax_c if cfg['type']=='class-incremental' else ax_d

    ax.errorbar(
        x, mean_per_task, yerr=std_per_task,
        fmt=marker, ms=10, color=colors[name],
        capsize=4, linestyle='none',
        label='_nolegend_'
    )
    ax.plot(x, m * x + b, linestyle='--', color=colors[name])

ax_c.set_xlim(*xlim_class)
ax_c.set_ylim(*ylim_class)
ax_c.set_xticks([1, 2, 3])
ax_d.set_xlim(*xlim_domain)
ax_d.set_ylim(*ylim_domain)
ax_d.set_xticks([1, 2, 3])

ax_c.set_title('Class-Incremental Tasks', fontsize=18)
ax_c.set_xlabel('Task Index', fontsize=20)
ax_c.set_ylabel('Neuronal Gain', fontsize=20)
ax_d.set_title('Domain-Incremental Tasks', fontsize=18)
ax_d.set_xlabel('Task Index', fontsize=20)
ax_d.set_ylabel('Neuronal Gain', fontsize=20)

for ax in (ax_c, ax_d):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', labelcolor='black')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

# --- legend ---
handles = [
    Line2D([], [], marker=marker, linestyle='none', color=colors[k], label=display_names[k])
    for k in datasets.keys()
]

handles_line1 = handles[:2]
handles_line2 = handles[2:]

fig.legend(
    handles=handles_line1,
    loc='upper center',
    ncol=len(handles_line1),
    frameon=False,
    fontsize=16,
    bbox_to_anchor=(0.5, 1.05)
)

fig.legend(
    handles=handles_line2,
    loc='upper center',
    ncol=len(handles_line2),
    frameon=False,
    fontsize=16,
    bbox_to_anchor=(0.5, 1.00)
)


plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


# Save
filename = 'fig_extra_gtaskComplexity'
#fig.savefig(f'{filename}.eps', format='eps', dpi=600, bbox_inches='tight') #eps
#fig.savefig(f'{filename}.svg', format='svg', dpi=600, bbox_inches='tight') #svg
#fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight') #png
