# -*- coding: utf-8 -*-
"""
Metrics
"""

import pickle
import numpy as np
import pandas as pd

def calc_ce_metrics_with_uncertainty(avg_hist, std_hist, ctx_iter):
    acc_mean = avg_hist.get("acc_test", [])
    acc_std  = std_hist.get("acc_test", [])
    num_tasks = len(acc_mean) // ctx_iter
    if num_tasks < 2:
        return {}, {}

    m_mean, m_std = {}, {}
    finals_mean, finals_std = [], []
    mins_mean, mins_std = [], []

    for t in range(num_tasks):
        block_mean = acc_mean[t*ctx_iter:(t+1)*ctx_iter]
        block_std  = acc_std[t*ctx_iter:(t+1)*ctx_iter]

        f_mean = block_mean[-1]
        f_std  = block_std[-1]
        m_mean[f"T{t+1}_final"] = f_mean
        m_std[f"T{t+1}_final"]  = f_std

        idx_min = int(np.argmin(block_mean))
        min_mean = block_mean[idx_min]
        min_std  = block_std[idx_min]
        m_mean[f"T{t+1}_min"] = min_mean
        m_std[f"T{t+1}_min"]  = min_std

        finals_mean.append(f_mean)
        finals_std.append(f_std)
        mins_mean.append(min_mean)
        mins_std.append(min_std)

    avg_min_acc = np.mean(mins_mean[:-1])
    std_min_acc = np.sqrt(np.sum(np.array(mins_std[:-1])**2)) / (num_tasks-1)
    m_mean["avg-min-ACC"] = avg_min_acc
    m_std["avg-min-ACC"]  = std_min_acc

    avg_acc = np.mean(finals_mean)
    std_acc = np.sqrt(np.sum(np.array(finals_std)**2)) / num_tasks
    m_mean["avg-ACC"] = avg_acc
    m_std["avg-ACC"]  = std_acc

    n = num_tasks
    wc_mean = (1/n)*finals_mean[-1] + (1 - 1/n)*avg_min_acc
    wc_std = np.sqrt((1/n)**2 * finals_std[-1]**2 +
                     (1 - 1/n)**2 * std_min_acc**2)
    m_mean["WC-ACC"] = wc_mean
    m_std["WC-ACC"]  = wc_std

    sg_list, sg_std_list = [], []
    for i in range(num_tasks-1):
        drop_mean = finals_mean[i] - mins_mean[i+1]
        drop_var  = finals_std[i]**2 + mins_std[i+1]**2
        sg_mean = drop_mean / finals_mean[i] if finals_mean[i] else np.nan
        sg_var = drop_var / finals_mean[i]**2 \
                 + (drop_mean**2 * finals_std[i]**2) / finals_mean[i]**4
        sg_std = np.sqrt(sg_var)

        key = f"SG_T{i+1}_to_T{i+2}"
        m_mean[key] = sg_mean
        m_std[key]  = sg_std
        sg_list.append(sg_mean)
        sg_std_list.append(sg_std)

    avg_sg = np.mean(sg_list)
    std_sg = np.sqrt(np.sum(np.array(sg_std_list)**2)) / len(sg_std_list)
    m_mean["Average_SG"] = avg_sg
    m_std["Average_SG"]  = std_sg

    return m_mean, m_std


# --------------------------------
# Load
# --------------------------------
with open('dataAvg_permMNIST.pkl', 'rb') as f:
    avg_perm, std_perm = pickle.load(f)
with open('dataAvg_rotMNIST.pkl',  'rb') as f:
    avg_rot,  std_rot  = pickle.load(f)
with open('dataAvg_splitMNIST.pkl','rb') as f:
    avg_split, std_split = pickle.load(f)
with open('dataAvg_splitCIFAR10.pkl','rb') as f:
    avg_cifar, std_cifar = pickle.load(f)
with open('dataAvg_domainCIFAR100.pkl', 'rb') as f:
    avg_domcifar, std_domcifar = pickle.load(f)
    
# --------------------------------
# Config
# --------------------------------
methods   = ['MSGD', 'ADAM', 'ENTROPY GAIN']
ctx_iters = {
    'permMNIST':    200,
    'rotMNIST':     400,
    'splitMNIST':   200,
    'splitCIFAR10': 400,
    'domainCIFAR100': 400,
}
datasets = {
    'permMNIST':    (avg_perm,  std_perm),
    'rotMNIST':     (avg_rot,   std_rot),
    'splitMNIST':   (avg_split, std_split),
    'splitCIFAR10': (avg_cifar, std_cifar),
    'domainCIFAR100': (avg_domcifar, std_domcifar)
}

# --------------------------------
# Metrics
# --------------------------------
metrics_tables = {}
for name, (avg_res, std_res) in datasets.items():
    all_metrics = {}
    for m in methods:
        mean_metrics, std_metrics = calc_ce_metrics_with_uncertainty(
            avg_res[m], std_res[m], ctx_iters[name]
        )
        formatted = {
            k: f"{mean_metrics[k]:.3f} ± {std_metrics[k]:.3f}"
            for k in mean_metrics
        }
        all_metrics[m] = formatted

    df = pd.DataFrame(all_metrics).T
    df.index.name = 'Method'
    metrics_tables[name] = df

metrics_permMNIST    = metrics_tables['permMNIST']
metrics_rotMNIST     = metrics_tables['rotMNIST']
metrics_splitMNIST   = metrics_tables['splitMNIST']
metrics_splitCIFAR10 = metrics_tables['splitCIFAR10']
metrics_domainCIFAR100 = metrics_tables['domainCIFAR100']


# New format
metrics_tables = {}
for name, (avg_res, std_res) in datasets.items():
    all_metrics = {}
    for m in methods:
        mean_metrics, std_metrics = calc_ce_metrics_with_uncertainty(
            avg_res[m], std_res[m], ctx_iters[name]
        )
        formatted = {k: f"{mean_metrics[k]:.3f} \u00B1 {std_metrics[k]:.3f}"
                     for k in mean_metrics}
        all_metrics[m] = formatted

    df = pd.DataFrame(all_metrics).T
    df.index.name = 'Method'
    df.rename(index={'ENTROPY GAIN': 'NGM-SGD'}, inplace=True)
    # Selección y renombrado de columnas para la tabla LaTeX
    df = df[['avg-ACC', 'avg-min-ACC', 'WC-ACC', 'Average_SG']]
    df = df.rename(columns={'Average_SG': 'avg-SG'})
    metrics_tables[name] = df

# --------------------------------
# LaTeX
# --------------------------------
combined = pd.concat(metrics_tables, names=['Dataset', 'Method'])

combined = combined.map(lambda x: x.replace(' ± ', r' $\pm$ '))

combined.rename(
    index={
        'permMNIST':      r'\shortstack{Permuted\\MNIST}',
        'rotMNIST':       r'\shortstack{Rotated\\MNIST}',
        'splitMNIST':     r'\shortstack{Split\\MNIST}',
        'splitCIFAR10':   r'\shortstack{Split\\CIFAR-10}',
        'domainCIFAR100': r'\shortstack{Domain\\CIFAR-100}'
    },
    level='Dataset',
    inplace=True
)

combined.rename(
    index={'ENTROPY GAIN': 'NGM-SGD'},
    level='Method',
    inplace=True
)

combined.index.names = [None, 'Method']

combined.rename(
    columns={
        'avg-ACC':     r'avg-ACC ($\uparrow$)',
        'avg-min-ACC': r'avg-min-ACC ($\uparrow$)',
        'WC-ACC':      r'WC-ACC ($\uparrow$)',
        'avg-SG':      r'avg-SG ($\downarrow$)'
    },
    inplace=True
)

def bold_best(group):
    for col in group.columns:
        # extrae la parte numérica antes del ' $'
        nums = group[col].str.split(r' \$').str[0].astype(float)
        # en avg-SG ($\downarrow$) elegimos el mínimo; en el resto, el máximo
        if col.startswith('avg-SG'):
            best_idx = nums.idxmin()
        else:
            best_idx = nums.idxmax()
        group.loc[best_idx, col] = f"\\textbf{{{group.loc[best_idx, col]}}}"
    return group

combined = combined.groupby(level=0, group_keys=False).apply(bold_best)

total_cols = combined.shape[1] + combined.index.nlevels  # 4 métricas + 2 niveles de índice
latex_table = combined.to_latex(
    multicolumn=True,
    multirow=True,
    index_names=False,                  # quita el título “Dataset”
    column_format='c' * total_cols,     # todas las columnas (y los índices) centrados
    escape=False,                       # para que \shortstack, \\ y $\pm$ funcionen
    caption='.',
    label='tab:ce_metrics'
)

print(latex_table)