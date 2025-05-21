# -*- coding: utf-8 -*-
"""
Preprocessing (old data)
"""

import pickle
import numpy as np

# 1. Load data
with open('data_permMNIST.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. List
methods = [
    m for m in data[0]['results'].keys()
    if m != 'BUMP GAIN'
]

# 3. Storage
avg_results = {}
std_results = {}

for method in methods:
    avg_results[method] = {}
    std_results[method] = {}

    
    all_metrics  = list(data[0]['results'][method].keys())
    gain_metrics = [m for m in all_metrics if m.lower().startswith('gain')]
    other_metrics = [m for m in all_metrics if m not in gain_metrics]

    
    for metric in other_metrics:
        arrays = [
            np.array(seed_exp['results'][method][metric])
            for seed_exp in data
        ]
        mat = np.vstack(arrays)  # shape (n_seeds, n_epochs)
        avg_results[method][metric] = mat.mean(axis=0).tolist()
        std_results[method][metric] = mat.std(axis=0).tolist()

    
    if gain_metrics:
        selected_gain = gain_metrics[2]

        
        gain_arrays = [
            np.array(seed_exp['results'][method][selected_gain])
            for seed_exp in data
        ]
        mat_gain = np.vstack(gain_arrays)  # (n_seeds, n_epochs)
        avg_results[method]['gain'] = mat_gain.mean(axis=0).tolist()
        std_results[method]['gain'] = mat_gain.std(axis=0).tolist()

data_avg = [avg_results, std_results]

# Save data
with open("dataAvg_permMNIST.pkl", "wb") as f:
    pickle.dump(data_avg, f)