# -*- coding: utf-8 -*-
"""
Split MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.distributions import Categorical
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# train dataset
train_ds = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
# test dataset
test_ds = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

## Network Model
# --- Layers ---
class GainModLayer(nn.Module):
    def __init__(self, input_dim, output_dim, g0=1.0, gamma=0.9, eta:float=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(output_dim))
        self.gains  = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        self.g0     = g0
        self.gamma  = gamma
        self.eta    = eta

    def decay(self, H_drive: float = 0.0):
        with torch.no_grad():
            self.gains.mul_(self.gamma).add_((1 - self.gamma) * self.g0).add_(self.eta * H_drive)
    
    def forward(self, x, H_drive: float = 0.0):
        if self.training: self.decay(H_drive)
        z = (x @ self.weight.t()) * self.gains + self.bias    
        return torch.relu(z)


class GainModOut(nn.Module):
    def __init__(self, input_dim, output_dim, g0=1.0, gamma=0.9, eta:float=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.05)
        self.bias   = nn.Parameter(torch.zeros(output_dim))
        self.gains  = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        self.g0     = g0
        self.gamma  = gamma
        self.eta    = eta
    
    def decay(self, H_drive: float = 0.0):
        with torch.no_grad():
            self.gains.mul_(self.gamma).add_((1 - self.gamma) * self.g0).add_(self.eta * H_drive)
     
    
    def forward(self, x, H_drive: float = 0.0):
        if self.training: self.decay(H_drive)
        z = (x @ self.weight.t()) * self.gains + self.bias    
        return z

# --- Network ---
class GainModNet(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=400, output_dim=10, g0=1.0, gamma=0.9, eta:float=0.0):
        super().__init__()
        self.layer1 = GainModLayer(input_dim,  hidden_dim, g0, gamma, eta)
        self.layer2 = GainModLayer(hidden_dim, hidden_dim, g0, gamma, eta)
        self.out    = GainModOut(hidden_dim,  output_dim, g0, gamma, eta)

    def forward(self, x, H_drive: float = 0.0):
        x  = x.view(x.size(0), -1)
        h1 = self.layer1(x, H_drive)
        h2 = self.layer2(h1, H_drive)
        y  = self.out(h2, H_drive)
        return y

## Continual evaluation
def continual_train(model, optimizer, criterion,contexts, batch, ctx_iter, rho_eval,train_ds, test_ds, device, mode):

    # --- DATA STORAGE ---
    hist = {k: [] for k in [
        "acc_train", "loss_train",
        "acc_test",  "loss_test",
        "w1", "w2", "w_out",
        "gain1", "gain2", "gain_out"
    ]}
    eval_loaders = {}
    step = 0
    H_prev = 0.0

    # --- CONTEXT LOOP ---
    for task_id, ctx in enumerate(contexts):
        print(f"\nContext {task_id+1}/{len(contexts)}: {ctx}")


        # --------------------- TRAIN LOADER --------------------------
        mask_train = torch.isin(train_ds.targets, torch.tensor(ctx))
        idx_train  = torch.nonzero(mask_train, as_tuple=False).squeeze()
        subset_train = Subset(train_ds, idx_train)
        sampler = RandomSampler(subset_train,replacement=True,num_samples=ctx_iter * batch)
        train_loader = DataLoader(subset_train, batch_size=batch, sampler=sampler)
        train_iter = iter(train_loader)

        # --------------------- TEST LOADER ---------------------------
        mask_test = torch.isin(test_ds.targets, torch.tensor([0,1]))
        idx_test  = torch.nonzero(mask_test, as_tuple=False).squeeze()
        subset_test = Subset(test_ds, idx_test)
        #eval_sampler = RandomSampler(subset_test, replacement=False, num_samples=1000)
        #test_loader  = DataLoader(subset_test, batch_size=batch, sampler=eval_sampler) 
        test_loader  = DataLoader(subset_test, batch_size=batch, shuffle=False) # all dataset each time
        eval_loaders[task_id] = test_loader

        # --------------------- TRAINING ------------------------------
        model.train()
        for _ in range(ctx_iter):
            
            inputs, targets = next(train_iter)
            inputs, targets = inputs.to(device), targets.to(device)

            # --- forward + backward + update ---
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1), H_prev)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # compute entropy 
            if mode == "gain-H":
                with torch.no_grad():
                    dist   = Categorical(logits=outputs)  # softmax
                    H_prev = dist.entropy().mean().item() # entropy

            # --- train metrics ---
            hist["acc_train"].append( (outputs.argmax(1) == targets).float().mean().item() )
            hist["loss_train"].append(loss.item())
            # --- weights and gains stats ---
            hist["w1"].append(model.layer1.weight.data.abs().mean().item())
            hist["w2"].append(model.layer2.weight.data.abs().mean().item())
            hist["w_out"].append(model.out.weight.data.abs().mean().item())
            hist["gain1"].append(model.layer1.gains.mean().item())
            hist["gain2"].append(model.layer2.gains.mean().item())
            hist["gain_out"].append(model.out.gains.mean().item())

            # ---------------- CONTINUAL EVAL --------------------------
            if step % rho_eval == 0:
                model.eval()
                total, correct, loss_sum = 0, 0, 0.0
                with torch.no_grad():
                    for ev_loader in eval_loaders.values():
                        for x_t, y_t in ev_loader:
                            x_t, y_t = x_t.to(device), y_t.to(device)
                            logits = model(x_t.view(x_t.size(0), -1), H_prev)
                            loss_ev = criterion(logits, y_t)

                            loss_sum += loss_ev.item() * y_t.size(0)
                            correct  += (logits.argmax(1) == y_t).sum().item()
                            total    += y_t.size(0)
                            
                # --- test metrics ---
                hist["acc_test"].append(100.0 * correct / total)
                hist["loss_test"].append(loss_sum / total)
                model.train()

            step += 1

    return hist

## Hyperparameters
batch = 128
lr_weights = 0.01
rho_eval = 1
ctx_iter = 200
contexts = [[0,1],[0,1,2,3], [0,1,2,3,4,5],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7,8,9]]
criterion = nn.CrossEntropyLoss()

## Simulation
def run_all_models_nSims(nSims, base_seed=0):
    all_results = []

    for sim in range(nSims):
        seed = base_seed + sim
        # --- Seeds ---
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n=== Sim {sim+1}/{nSims}, seed={seed} ===")

        # --- ENTROPY GAIN MODEL ---
        print("ENTROPY GAIN MODEL")
        hgainModel = GainModNet(input_dim=28*28, hidden_dim=400, output_dim=10,
                                 g0=1.0, gamma=0.9, eta=0.4).to(device)
        hgainOptimizer = optim.SGD([
            {'params': [param for name, param in hgainModel.named_parameters() if 'weight' in name],
             'lr': lr_weights},
            {'params': [param for name, param in hgainModel.named_parameters() if 'weight' not in name],
             'lr': 0},
        ], momentum=0.0)
        hgain_hist = continual_train(hgainModel, hgainOptimizer, criterion,
                                     contexts, batch, ctx_iter, rho_eval,
                                     train_ds, test_ds, device, mode="gain-H")


        # --- VANILLA ADAM ---
        print("VANILLA ADAM")
        adamModel = GainModNet(input_dim=28*28, hidden_dim=400, output_dim=10,
                                g0=1.0, gamma=0.0, eta=0.0).to(device)
        adamOptimizer = optim.Adam([
            {'params': [param for name, param in adamModel.named_parameters() if 'weight' in name],
             'lr': lr_weights},
            {'params': [param for name, param in adamModel.named_parameters() if 'weight' not in name],
             'lr': 0},
        ])
        adam_hist = continual_train(adamModel, adamOptimizer, criterion,
                                    contexts, batch, ctx_iter, rho_eval,
                                    train_ds, test_ds, device, mode="no-gain")

        # --- MOMENTUM SGD ---
        print("MOMENTUM SGD")
        msgdModel = GainModNet(input_dim=28*28, hidden_dim=400, output_dim=10,
                                g0=1.0, gamma=0.0, eta=0.0).to(device)
        msgdOptimizer = optim.SGD([
            {'params': [param for name, param in msgdModel.named_parameters() if 'weight' in name],
             'lr': lr_weights},
            {'params': [param for name, param in msgdModel.named_parameters() if 'weight' not in name],
             'lr': 0},
        ], momentum=0.9)
        msgd_hist = continual_train(msgdModel, msgdOptimizer, criterion,
                                    contexts, batch, ctx_iter, rho_eval,
                                    train_ds, test_ds, device, mode="no-gain")

        # Recopilar resultados de esta simulación
        results = {
            "ENTROPY GAIN": hgain_hist,
            "ADAM":       adam_hist,
            "MSGD":       msgd_hist
        }
        all_results.append({"seed": seed, "results": results})

    return all_results

## Run
all_sims = run_all_models_nSims(nSims=5, base_seed=42)

