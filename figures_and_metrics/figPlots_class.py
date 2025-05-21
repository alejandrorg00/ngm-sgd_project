# -*- coding: utf-8 -*-
"""
Class-incremental plots
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='sans-serif', serif=['Arial'])
plt.rc('axes', edgecolor='black', linewidth=1)
plt.rc('xtick', color='slategray')
plt.rc('ytick', color='slategray')
plt.rc('grid', color='slategray')

# 1. Load averaged results for MNIST (left) and CIFAR-10 (right)
with open('dataAvg_splitMNIST.pkl', 'rb') as f:
    data_L = pickle.load(f)
with open('dataAvg_splitCIFAR10.pkl', 'rb') as f:
    data_R = pickle.load(f)

methods = ['ADAM', 
           'MSGD', 
           'ENTROPY GAIN']
colors  = {'ADAM':'mediumseagreen',
           'MSGD':'royalblue',
           'ENTROPY GAIN':'darkorange'}

# 2. Epoch arrays
n_L = len(data_L[0][methods[0]]['acc_test'])
n_R = len(data_R[0][methods[0]]['acc_test'])
it_L = np.arange(1, n_L+1)
it_R = np.arange(1, n_R+1)

# 3. Context‐switch parameters
interval_L = 200
interval_R = 400
num_tasks  = 5

def draw_task_lines(ax, n_epochs, interval):
    for b in range(0, (num_tasks+1)*interval, interval):
        ax.axvline(b, color='slategray', linestyle='-', linewidth=0.75, zorder=0)

def label_tasks_under(ax, n_epochs, interval):
    for i in range(num_tasks):
        x = i*interval + interval/2
        ax.text(
            x, -0.025, f"Task {i+1}",
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=14, fontweight='normal', fontstyle='normal'
        )

# 4. Build figure with a 4×2 grid, allocate subplots:
fig = plt.figure(figsize=(12, 5), constrained_layout=True)

ax_acc_R = fig.add_subplot(4, 2, (1, 3))
ax_acc_L = fig.add_subplot(4, 2, (2, 4))
ax_loss_R = fig.add_subplot(4, 2, 5)
ax_loss_L = fig.add_subplot(4, 2, 6)
ax_gain_L = fig.add_subplot(4, 2, 7)
ax_gain_R = fig.add_subplot(4, 2, 8)

handles = []

# === CIFAR-10 Accuracy ===
draw_task_lines(ax_acc_R, n_R, interval_R)
for m in methods:
    mu = np.array(data_R[0][m]['acc_test'])
    sd = np.array(data_R[1][m]['acc_test'])
    line, = ax_acc_R.plot(it_R, mu, color=colors[m], zorder=2)
    ax_acc_R.fill_between(it_R, mu-sd, mu+sd, color=colors[m], alpha=0.3, zorder=1)
    handles.append(line)
ax_acc_R.set_title("Split CIFAR-10", fontsize=15, fontweight='normal')
ax_acc_R.set_ylim(26, 100)
ax_acc_R.grid(axis='y', linestyle='-', linewidth=0.25)
ax_acc_R.set_ylabel('% Task 1 \nAccuracy', fontsize=14, fontweight='normal')
# ticks every 400, but hide labels here
ax_acc_R.set_xticks(np.arange(0, n_R+1, interval_R))
ax_acc_R.tick_params(labelbottom=False)
label_tasks_under(ax_acc_R, n_R, interval_R)
ax_acc_R.spines['top'].set_visible(False)
ax_acc_R.spines['right'].set_visible(False)

# === MNIST Accuracy ===
draw_task_lines(ax_acc_L, n_L, interval_L)
for m in methods:
    mu = np.array(data_L[0][m]['acc_test'])
    sd = np.array(data_L[1][m]['acc_test'])
    ax_acc_L.plot(it_L, mu, color=colors[m], zorder=2)
    ax_acc_L.fill_between(it_L, mu-sd, mu+sd, color=colors[m], alpha=0.3, zorder=1)
ax_acc_L.set_title("Split MNIST",fontsize=15, fontweight='normal')
ax_acc_L.set_ylim(91, 100)
ax_acc_L.grid(axis='y', linestyle='-', linewidth=0.25)
# ticks every 200, but hide labels here
ax_acc_L.set_xticks(np.arange(0, n_L+1, interval_L))
ax_acc_L.tick_params(labelbottom=False)
label_tasks_under(ax_acc_L, n_L, interval_L)
ax_acc_L.spines['top'].set_visible(False)
ax_acc_L.spines['right'].set_visible(False)

# === CIFAR-10 Loss ===
draw_task_lines(ax_loss_R, n_R, interval_R)
for m in methods:
    mu = np.array(data_R[0][m]['loss_test'])
    sd = np.array(data_R[1][m]['loss_test'])
    ax_loss_R.plot(it_R, mu, color=colors[m], zorder=2)
    ax_loss_R.fill_between(it_R, mu-sd, mu+sd, color=colors[m], alpha=0.3, zorder=1)
ax_loss_R.set_ylim(0, 5.0)
ax_loss_R.grid(axis='y', linestyle='-', linewidth=0.25)
ax_loss_R.set_ylabel('Task 1 \nLoss', fontsize=14, fontweight='normal')
# ticks every 400, labels off
ax_loss_R.set_xticks(np.arange(0, n_R+1, interval_R))
ax_loss_R.tick_params(labelbottom=False)
ax_loss_R.spines['top'].set_visible(False)
ax_loss_R.spines['right'].set_visible(False)

# === MNIST Loss ===
draw_task_lines(ax_loss_L, n_L, interval_L)
for m in methods:
    mu = np.array(data_L[0][m]['loss_test'])
    sd = np.array(data_L[1][m]['loss_test'])
    ax_loss_L.plot(it_L, mu, color=colors[m], zorder=2)
    ax_loss_L.fill_between(it_L, mu-sd, mu+sd, color=colors[m], alpha=0.3, zorder=1)
ax_loss_L.set_ylim(0, 1.25)
ax_loss_L.grid(axis='y', linestyle='-', linewidth=0.25)
# ticks every 200, labels off
ax_loss_L.set_xticks(np.arange(0, n_L+1, interval_L))
ax_loss_L.tick_params(labelbottom=False)
ax_loss_L.spines['top'].set_visible(False)
ax_loss_L.spines['right'].set_visible(False)

# === CIFAR-10 Gain (left gain panel) ===
draw_task_lines(ax_gain_L, n_R, interval_R)
m = 'ENTROPY GAIN'
mu = np.array(data_R[0][m]['gain'])
sd = np.array(data_R[1][m]['gain'])
ax_gain_L.plot(it_R, mu, color=colors[m], zorder=2)
ax_gain_L.fill_between(it_R, mu-sd, mu+sd, color=colors[m], alpha=0.3, zorder=1)
ax_gain_L.set_ylim(1.0, 3.25)
ax_gain_L.set_xlabel("Iteration", fontsize=14, fontweight='normal')
ax_gain_L.grid(axis='y', linestyle='-', linewidth=0.25)
ax_gain_L.set_ylabel('Neuronal \nGain', fontsize=14, fontweight='normal')
# ticks every 400, labels on
ax_gain_L.set_xticks(np.arange(0, n_R+1, interval_R))
ax_gain_L.tick_params(labelbottom=True)
ax_gain_L.spines['top'].set_visible(False)
ax_gain_L.spines['right'].set_visible(False)

# === MNIST Gain (right gain panel) ===
draw_task_lines(ax_gain_R, n_L, interval_L)
mu = np.array(data_L[0][m]['gain'])
sd = np.array(data_L[1][m]['gain'])
ax_gain_R.plot(it_L, mu, color=colors[m], zorder=2)
#ax_gain_R.set_ylim(1.0, 3.25)
ax_gain_R.fill_between(it_L, mu-sd, mu+sd, color=colors[m], alpha=0.3, zorder=1)
ax_gain_R.set_xlabel("Iteration", fontsize=14, fontweight='normal')
ax_gain_R.grid(axis='y', linestyle='-', linewidth=0.25)
# ticks every 200, labels on
ax_gain_R.set_xticks(np.arange(0, n_L+1, interval_L))
ax_gain_R.tick_params(labelbottom=True)
ax_gain_R.spines['top'].set_visible(False)
ax_gain_R.spines['right'].set_visible(False)

# 5. Global legend
# after you’ve built `handles` in the same order as `methods`
label_map = {
    'ADAM': 'ADAM',
    'MSGD': 'MSGD',
    'ENTROPY GAIN': 'NGM-SDG'
}
legend_labels = [label_map[m] for m in methods]

fig.legend(
    handles,
    legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),   # move higher
    ncol=len(legend_labels),
    frameon=False,
    prop={'size':16,'weight':'normal'}
)

for ax in [ax_acc_L, ax_acc_R,
           ax_loss_L, ax_loss_R,
           ax_gain_L, ax_gain_R]:

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
filename = 'fig_split'
#fig.savefig(f'{filename}.eps', format='eps', dpi=600, bbox_inches='tight') #eps
#fig.savefig(f'{filename}.svg', format='svg', dpi=600, bbox_inches='tight') #svg
#fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight') #png