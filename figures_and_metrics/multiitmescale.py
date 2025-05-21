# -*- coding: utf-8 -*-
"""
Multitimescale plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rc('font', family='sans-serif', serif=['Arial'])
plt.rc('axes', edgecolor='black')
plt.rc('xtick', color='black')
plt.rc('ytick', color='black')
plt.rc('grid', color='black')

# --- First Simulation Data ---
n_steps1 = 300
x = 1.0
T1 = np.ones(n_steps1)
T1[144:155] = -1
T1[154:] = 1

alpha_fast, alpha_slow, fast_decay = 0.2, 0.1, 0.9
w_fast = w_slow = w0 = 0.0
alpha, g0, g_decay, g_alpha = 0.1, 1.0, 0.9, 0.7
w = g = g0

y0_log = np.zeros(n_steps1)
y_log = np.zeros(n_steps1)
y_mod_log = np.zeros(n_steps1)

for t in range(n_steps1):
    # Slow LR
    y0 = w0 * x
    err0 = T1[t] - y0
    y0_log[t] = y0
    w0 += alpha_slow * err0 * x

    # Fast-slow
    w_total = w_fast + w_slow
    y_log[t] = w_total
    err1 = T1[t] - w_total
    w_fast = fast_decay * w_fast + alpha_fast * err1 * x
    w_slow += alpha_slow * err1 * x

    # Gain-modulated
    y_mod = g * w * x
    y_mod_log[t] = y_mod
    err2 = T1[t] - y_mod
    w += alpha * g * err2 * x
    g = g_decay * g + (1 - g_decay) * g0 + g_alpha * abs(w * err2 * x)

# --- Second Simulation Statistics ---
n_steps2 = 100
T2 = np.zeros(n_steps2)
T2[44:] = 1
num_runs = 50

alpha, g0, g_decay, g_alpha = 0.1, 1.0, 0.5, 1.0
noise_std_g, noise_std_w = 0.05, 0.025

G = np.zeros((num_runs, n_steps2))
W0 = np.zeros((num_runs, n_steps2))
Wg0 = np.zeros((num_runs, n_steps2))
Wm = np.zeros((num_runs, n_steps2))

for run in range(num_runs):
    g = g0
    w = 0.0
    for t in range(n_steps2):
        g_noisy = g + noise_std_g * np.random.randn()
        w_noisy = w + noise_std_w * np.random.randn()
        err = T2[t] - (g_noisy * w_noisy * x)
        G[run, t] = g_noisy
        W0[run, t] = g0 * w_noisy
        Wg0[run, t] = (g_noisy - g0) * w_noisy
        Wm[run, t] = g_noisy * w_noisy
        w += alpha * g_noisy * err * x
        g = g_decay * g + (1 - g_decay) * g0 + g_alpha * abs(w * err * x)

t2 = np.arange(1, n_steps2 + 1)
G_mean, G_std = G.mean(axis=0), G.std(axis=0)
W0_mean, W0_std = W0.mean(axis=0), W0.std(axis=0)
Wg0_mean, Wg0_std = Wg0.mean(axis=0), Wg0.std(axis=0)
Wm_mean, Wm_std = Wm.mean(axis=0), Wm.std(axis=0)


# --- Plotting ---
fig = plt.figure(figsize=(18, 7), facecolor='w')
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=0.4)

# Panel 1: Neuronal Gain with noise
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(t2, G_mean - G_std, G_mean + G_std, color='limegreen', alpha=0.3)
ax1.axhline(g0, linestyle='--', color='slategray', linewidth=4)
ax1.plot(t2, G_mean, color='limegreen', linewidth=3)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.spines['top'].set_visible(False); 
ax1.spines['bottom'].set_visible(False);
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(5); ax1.spines['left'].set_linewidth(5)
ax1.set_title('A)', loc='left', fontsize=42, fontweight='bold', x=-0.25)
ax1.set_ylabel('Neuronal\nGain', fontsize=30)

# Panel 2 & 4: Comparison Fast–Slow vs Gain-modulated
ax2 = fig.add_subplot(gs[:, 1])
ax2.plot(y0_log, linestyle='-', linewidth=3, label='Slow weight', color='indigo')
ax2.plot(y_log, linewidth=3, label='Fast-slow\nweights', color='royalblue')
ax2.plot(y_mod_log, linewidth=3, label='Gain-modulated\nweights', color='darkorange')
ax2.plot(T1, linestyle='None', marker='.', markersize=10, label='Target', color='k')
ax2.set_xlim(100, 200); ax2.set_ylim(-1.5, 2.5)
ax2.set_xticks(np.arange(110, 201, 20)); ax2.set_yticks([-1.25, 0.75, 1.75])
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_linewidth(5); ax2.spines['left'].set_linewidth(5)
ax2.xaxis.set_tick_params(width=2.5, length=6, direction='out')
ax2.yaxis.set_tick_params(width=2.5, length=6, direction='out')
ax2.tick_params(axis='both', labelsize=16)
ax2.set_xlabel('Timesteps', fontsize=30)
ax2.set_ylabel('Output', fontsize=30)
ax2.set_title('B)', loc='left', fontsize=42, fontweight='bold', x=-0.25)
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=24)

# Panel 3: Weight decomposition with bands ±1σ
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(t2, Wg0_mean - Wg0_std, Wg0_mean + Wg0_std, color='red', alpha=0.2)
ax3.fill_between(t2, W0_mean - W0_std, W0_mean + W0_std, color='brown', alpha=0.2)
ax3.fill_between(t2, Wm_mean - Wm_std, Wm_mean + Wm_std, color='darkorange', alpha=0.2)
ax3.plot(t2, Wg0_mean, color='red', linewidth=3, label='w_{fast}=(g-g_0)w')
ax3.plot(t2, W0_mean, color='brown', linewidth=3, label='w_{slow}=g_0w')
ax3.plot(t2, Wm_mean, color='darkorange', linewidth=3, label='W_{eff}=gw')
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_xlim(0, n_steps2 + 1); ax3.set_ylim(-0.5, 1.25)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_linewidth(5); ax3.spines['left'].set_linewidth(5)
ax3.set_xlabel('Time',   fontsize=30, labelpad=25) 
ax3.set_ylabel('Weight', fontsize=30, labelpad=20) 


plt.tight_layout()
plt.show()

# Save
filename = 'fig_multitimescale'
#fig.savefig(f'{filename}.eps', format='eps', dpi=600, bbox_inches='tight') #eps
#fig.savefig(f'{filename}.svg', format='svg', dpi=600, bbox_inches='tight') #svg
#fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight') #png