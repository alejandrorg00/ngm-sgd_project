# -*- coding: utf-8 -*-
"""
Training plots
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Config ---
methods = ['ADAM', 'MSGD', 'ENTROPY GAIN']
legend_labels = {'ADAM':'ADAM', 'MSGD':'MSGD', 'ENTROPY GAIN':'NMG-SDG'}
colors  = {
    'ADAM':         'mediumseagreen',
    'MSGD':         'royalblue',
    'ENTROPY GAIN': 'darkorange'
}

datasets = {
    'splitMNIST':      ('dataAvg_splitMNIST.pkl',     'Split MNIST'),
    'splitCIFAR10':    ('dataAvg_splitCIFAR10.pkl',   'Split CIFAR-10'),
    'permMNIST':       ('dataAvg_permMNIST.pkl',      'Permuted MNIST'),
    'rotMNIST':        ('dataAvg_rotMNIST.pkl',       'Rotated MNIST'),
    'domainCIFAR100':  ('dataAvg_domainCIFAR100.pkl', 'Domain CIFAR-100'),
}

# --- GridSpec ---
plt.rc('axes', titlesize=15, labelsize=16)
fig = plt.figure(figsize=(12, 6))
gs  = fig.add_gridspec(2, 6)

# Top row
ax1 = fig.add_subplot(gs[0, 0:3])  # splitMNIST
ax2 = fig.add_subplot(gs[0, 3:6])  # splitCIFAR10

# Bottom row
ax3 = fig.add_subplot(gs[1, 0:2])  # permMNIST
ax4 = fig.add_subplot(gs[1, 2:4])  # rotMNIST
ax5 = fig.add_subplot(gs[1, 4:6])  # domainCIFAR100

axes = [ax1, ax2, ax3, ax4, ax5]
keys  = ['splitMNIST', 'splitCIFAR10', 'permMNIST', 'rotMNIST', 'domainCIFAR100']

# --- Plot ---
for ax, key in zip(axes, keys):
    filename, display_title = datasets[key]
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    for method in methods:
        acc = np.array(data[0][method]['acc_train'])
        sd  = np.array(data[1][method]['acc_train'])
        x = np.arange(len(acc))

        ax.plot(x, acc, label=legend_labels[method], color=colors[method], lw=2)
        ax.fill_between(x,
                        acc - sd,
                        acc + sd,
                        color=colors[method],
                        alpha=0.3)

    ax.set_title(display_title)
    ax.set_xlabel('Iteration')
    if ax in (ax1, ax3):
        ax.set_ylabel('Training Accuracy')
    ax.grid(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='both', labelcolor='black')

# --- legend ---
handles = [Line2D([], [], color=colors[m], lw=2, label=legend_labels[m]) 
           for m in methods]
fig.legend(handles=handles,
           loc='upper center',
           fontsize = 16,
           ncol=len(methods),
           frameon=False,
           bbox_to_anchor=(0.5, 1.025))

fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# Save
filename = 'fig_extra_training'
#fig.savefig(f'{filename}.eps', format='eps', dpi=600, bbox_inches='tight') #eps
#fig.savefig(f'{filename}.svg', format='svg', dpi=600, bbox_inches='tight') #svg
#fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight') #png