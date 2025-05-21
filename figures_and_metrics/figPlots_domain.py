# -*- coding: utf-8 -*-
"""
Domain-incremental plots
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif=['Arial'])
plt.rc('axes', edgecolor='black', linewidth=1)
plt.rc('xtick', color='slategray')
plt.rc('ytick', color='slategray')
plt.rc('grid', color='slategray')


# 1. Load averaged results for permuted MNIST (left), rotated MNIST (mid) and CIFAR-100 (right)
with open('dataAvg_permMNIST.pkl', 'rb') as f:
    data_L = pickle.load(f)
with open('dataAvg_rotMNIST.pkl', 'rb') as f:
    data_M = pickle.load(f)
with open('dataAvg_domainCIFAR100.pkl', 'rb') as f:
    data_R = pickle.load(f)

methods = ['ADAM', 'MSGD', 'ENTROPY GAIN']
colors  = {'ADAM': 'mediumseagreen',
           'MSGD': 'royalblue',
           'ENTROPY GAIN': 'darkorange'}

# 2. Epoch arrays
n_L = len(data_L[0][methods[0]]['acc_test'])
n_M = len(data_M[0][methods[0]]['acc_test'])
n_R = len(data_R[0][methods[0]]['acc_test'])
it_L = np.arange(1, n_L + 1)
it_M = np.arange(1, n_M + 1)
it_R = np.arange(1, n_R + 1)

# 3. Context-switch parameters
interval_L = 200
interval_M = 400
interval_R = 400
num_tasks  = 3

def draw_task_lines(ax, interval):
    for b in range(0, (num_tasks + 1) * interval, interval):
        ax.axvline(b, color='slategray', linestyle='-', linewidth=0.75, zorder=0)

def label_tasks_under(ax, interval):
    for i in range(num_tasks):
        x = i * interval + interval / 2
        ax.text(
            x, -0.025, f"Task {i+1}",
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=14, fontweight='normal', fontstyle='normal'
        )

# 4. Build figure with an 8×3 grid
fig = plt.figure(figsize=(12, 5), constrained_layout=True)
gs  = fig.add_gridspec(8, 3)

# === permuted MNIST Accuracy (left accuracy panel) ===
ax_acc_L = fig.add_subplot(gs[0:4, 0])
draw_task_lines(ax_acc_L, interval_L)
for m in methods:
    mu = np.array(data_L[0][m]['acc_test'])
    sd = np.array(data_L[1][m]['acc_test'])
    ax_acc_L.plot(it_L, mu, color=colors[m], zorder=2)
    ax_acc_L.fill_between(it_L, mu - sd, mu + sd, color=colors[m], alpha=0.3, zorder=1)
ax_acc_L.set_title('Permuted MNIST', fontsize=15, fontweight='normal')
ax_acc_L.set_ylim(78, 98)
ax_acc_L.grid(axis='y', linestyle='-', linewidth=0.25)
ax_acc_L.set_xticks(np.arange(0, n_L + 1, interval_L))
ax_acc_L.tick_params(labelbottom=False)
label_tasks_under(ax_acc_L, interval_L)
ax_acc_L.set_ylabel('% Task 1\nAccuracy', fontsize=14)
ax_acc_L.spines['top'].set_visible(False)
ax_acc_L.spines['right'].set_visible(False)

# === rotated MNIST Accuracy (middle accuracy panel) ===
ax_acc_M = fig.add_subplot(gs[0:4, 1])
draw_task_lines(ax_acc_M, interval_M)
for m in methods:
    mu = np.array(data_M[0][m]['acc_test'])
    sd = np.array(data_M[1][m]['acc_test'])
    ax_acc_M.plot(it_M, mu, color=colors[m], zorder=2)
    ax_acc_M.fill_between(it_M, mu - sd, mu + sd, color=colors[m], alpha=0.3, zorder=1)
ax_acc_M.set_title('Rotated MNIST', fontsize=15, fontweight='normal')
ax_acc_M.set_ylim(78, 98)
ax_acc_M.grid(axis='y', linestyle='-', linewidth=0.25)
ax_acc_M.set_xticks(np.arange(0, n_M + 1, interval_M))
ax_acc_M.tick_params(labelbottom=False)
label_tasks_under(ax_acc_M, interval_M)
ax_acc_M.spines['top'].set_visible(False)
ax_acc_M.spines['right'].set_visible(False)

# === CIFAR-100 Accuracy (right accuracy panel) ===
ax_acc_R = fig.add_subplot(gs[0:4, 2])
draw_task_lines(ax_acc_R, interval_R)
for m in methods:
    mu = np.array(data_R[0][m]['acc_test'])
    sd = np.array(data_R[1][m]['acc_test'])
    ax_acc_R.plot(it_R, mu, color=colors[m], zorder=2)
    ax_acc_R.fill_between(it_R, mu - sd, mu + sd, color=colors[m], alpha=0.3, zorder=1)
ax_acc_R.set_title('Domain CIFAR-100', fontsize=15, fontweight='normal')
ax_acc_R.set_ylim(8, 58)
ax_acc_R.grid(axis='y', linestyle='-', linewidth=0.25)
ax_acc_R.set_xticks(np.arange(0, n_R + 1, interval_R))
ax_acc_R.tick_params(labelbottom=False)
label_tasks_under(ax_acc_R, interval_R)
ax_acc_R.spines['top'].set_visible(False)
ax_acc_R.spines['right'].set_visible(False)

# === permuted MNIST Loss (left loss panel) ===
ax_loss_L = fig.add_subplot(gs[4:6, 0])
draw_task_lines(ax_loss_L, interval_L)
for m in methods:
    mu = np.array(data_L[0][m]['loss_test'])
    sd = np.array(data_L[1][m]['loss_test'])
    ax_loss_L.plot(it_L, mu, color=colors[m], zorder=2)
    ax_loss_L.fill_between(it_L, mu - sd, mu + sd, color=colors[m], alpha=0.3, zorder=1)
ax_loss_L.set_ylim(0, 1.25)
ax_loss_L.grid(axis='y', linestyle='-', linewidth=0.25)
ax_loss_L.set_xticks(np.arange(0, n_L + 1, interval_L))
ax_loss_L.tick_params(labelbottom=False)
ax_loss_L.set_ylabel('Task 1\nLoss', fontsize=14)
ax_loss_L.spines['top'].set_visible(False)
ax_loss_L.spines['right'].set_visible(False)

# === rotated MNIST Loss (middle loss panel) ===
ax_loss_M = fig.add_subplot(gs[4:6, 1])
draw_task_lines(ax_loss_M, interval_M)
for m in methods:
    mu = np.array(data_M[0][m]['loss_test'])
    sd = np.array(data_M[1][m]['loss_test'])
    ax_loss_M.plot(it_M, mu, color=colors[m], zorder=2)
    ax_loss_M.fill_between(it_M, mu - sd, mu + sd, color=colors[m], alpha=0.3, zorder=1)
ax_loss_M.set_ylim(0, 1.25)
ax_loss_M.grid(axis='y', linestyle='-', linewidth=0.25)
ax_loss_M.set_xticks(np.arange(0, n_M + 1, interval_M))
ax_loss_M.tick_params(labelbottom=False)
ax_loss_M.spines['top'].set_visible(False)
ax_loss_M.spines['right'].set_visible(False)

# === CIFAR-100 Loss (right loss panel) ===
ax_loss_R = fig.add_subplot(gs[4:6, 2])
draw_task_lines(ax_loss_R, interval_R)
for m in methods:
    mu = np.array(data_R[0][m]['loss_test'])
    sd = np.array(data_R[1][m]['loss_test'])
    ax_loss_R.plot(it_R, mu, color=colors[m], zorder=2)
    ax_loss_R.fill_between(it_R, mu - sd, mu + sd, color=colors[m], alpha=0.3, zorder=1)
ax_loss_R.set_ylim(-2, 43.0)
ax_loss_R.grid(axis='y', linestyle='-', linewidth=0.25)
ax_loss_R.set_xticks(np.arange(0, n_R + 1, interval_R))
ax_loss_R.tick_params(labelbottom=False)
ax_loss_R.spines['top'].set_visible(False)
ax_loss_R.spines['right'].set_visible(False)

# === permuted MNIST Gain (left gain panel) ===
ax_gain_L = fig.add_subplot(gs[6:8, 0])
draw_task_lines(ax_gain_L, interval_L)
mu = np.array(data_L[0]['ENTROPY GAIN']['gain'])
sd = np.array(data_L[1]['ENTROPY GAIN']['gain'])
ax_gain_L.plot(it_L, mu, color=colors['ENTROPY GAIN'], zorder=2)
ax_gain_L.fill_between(it_L, mu - sd, mu + sd, color=colors['ENTROPY GAIN'], alpha=0.3, zorder=1)
ax_gain_L.set_ylim(1.0, 5.5)
ax_gain_L.grid(axis='y', linestyle='-', linewidth=0.25)
ax_gain_L.set_xticks(np.arange(0, n_L + 1, interval_L))
ax_gain_L.tick_params(labelbottom=True)
ax_gain_L.set_ylabel('Neuronal\nGain', fontsize=14)
ax_gain_L.set_xlabel('Iteration', fontsize=14)
ax_gain_L.spines['top'].set_visible(False)
ax_gain_L.spines['right'].set_visible(False)

# === rotated MNIST Gain (middle gain panel) ===
ax_gain_M = fig.add_subplot(gs[6:8, 1])
draw_task_lines(ax_gain_M, interval_M)
mu = np.array(data_M[0]['ENTROPY GAIN']['gain'])
sd = np.array(data_M[1]['ENTROPY GAIN']['gain'])
ax_gain_M.plot(it_M, mu, color=colors['ENTROPY GAIN'], zorder=2)
ax_gain_M.fill_between(it_M, mu - sd, mu + sd, color=colors['ENTROPY GAIN'], alpha=0.3, zorder=1)
ax_gain_M.set_ylim(1.0, 5.5)
ax_gain_M.grid(axis='y', linestyle='-', linewidth=0.25)
ax_gain_M.set_xticks(np.arange(0, n_M + 1, interval_M))
ax_gain_M.tick_params(labelbottom=True)
ax_gain_M.set_xlabel('Iteration', fontsize=14)
ax_gain_M.spines['top'].set_visible(False)
ax_gain_M.spines['right'].set_visible(False)

# === CIFAR-100 Gain (right gain panel) ===
ax_gain_R = fig.add_subplot(gs[6:8, 2])
draw_task_lines(ax_gain_R, interval_R)
mu = np.array(data_R[0]['ENTROPY GAIN']['gain'])
sd = np.array(data_R[1]['ENTROPY GAIN']['gain'])
ax_gain_R.plot(it_R, mu, color=colors['ENTROPY GAIN'], zorder=2)
ax_gain_R.fill_between(it_R, mu - sd, mu + sd, color=colors['ENTROPY GAIN'], alpha=0.3, zorder=1)
ax_gain_R.set_ylim(1.0, 5.5)
ax_gain_R.grid(axis='y', linestyle='-', linewidth=0.25)
ax_gain_R.set_xticks(np.arange(0, n_R + 1, interval_R))
ax_gain_R.tick_params(labelbottom=True)
ax_gain_R.set_xlabel('Iteration', fontsize=14)
ax_gain_R.spines['top'].set_visible(False)
ax_gain_R.spines['right'].set_visible(False)

# 5. Global legend (using handles from the first accuracy panel)
handles = [plt.Line2D([], [], color=colors[m]) for m in methods]
label_map = {'ADAM': 'ADAM', 'MSGD': 'MSGD', 'ENTROPY GAIN': 'NGM-SDG'}
legend_labels = [label_map[m] for m in methods]
fig.legend(
    handles,
    legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),
    ncol=len(legend_labels),
    frameon=False,
    prop={'size': 16}
)

for ax in [ax_acc_L, ax_acc_M, ax_acc_R,
           ax_loss_L, ax_loss_M, ax_loss_R,
           ax_gain_L, ax_gain_M, ax_gain_R]:

    ax.tick_params(axis='y',
                   which='both',
                   length=0,          
                   labelcolor='black')

    ax.tick_params(axis='x',
                   which='both',
                   width=0.75,
                   labelcolor='black')

plt.show()

# Save
filename = 'fig_domain'
#fig.savefig(f'{filename}.eps', format='eps', dpi=600, bbox_inches='tight') #eps
#fig.savefig(f'{filename}.svg', format='svg', dpi=600, bbox_inches='tight') #svg
#fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight') #png