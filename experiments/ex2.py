# -*- coding: utf-8 -*-
"""
Split CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.distributions import Categorical
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Load data
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])
# train dataset
train_ds = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
# test dataset
test_ds = datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

## Network Model
# --- Conv ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False)

# --- Slim ResNet ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf,
                 global_pooling, input_size):
        super().__init__()
        self.global_pooling = global_pooling

        self.in_planes = nf
        self.conv1 = conv3x3(3, nf)
        self.bn1   = nn.BatchNorm2d(nf)
        self.layer1 = self._make_layer(block, nf,     num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        input_size = tuple(input_size[-3:])
        if input_size == (3,32,32):
            self.feature_size = 160  if global_pooling else 2560
        elif input_size == (3,84,84):
            self.feature_size = 640  if global_pooling else 19360
        elif input_size == (3,96,96):
            self.feature_size = 1440 if global_pooling else 23040
        else:
            raise ValueError(f"Input size no reconocido: {input_size}")

    def _make_layer(self, block, planes, n_blocks, stride):
        strides = [stride] + [1]*(n_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.global_pooling:
            out = avg_pool2d(out, 4)
        return out.view(out.size(0), -1)

def ResNet18feat(input_size, nf=20, global_pooling=True):
    return ResNet(BasicBlock, [2,2,2,2], nf, global_pooling, input_size)

# --- Head ---
class GainModLinear(nn.Module):
    def __init__(self, in_dim, out_dim,
                 g0=1.0, gamma=0.9, eta=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.05)
        self.bias   = nn.Parameter(torch.zeros(out_dim))
        self.gains  = nn.Parameter(torch.ones(out_dim) * g0,
                                   requires_grad=False)
        self.g0, self.gamma, self.eta = g0, gamma, eta

    def decay(self, H_drive: float = 0.0):
        with torch.no_grad():
            self.gains.mul_(self.gamma) \
                      .add_((1 - self.gamma) * self.g0) \
                      .add_(self.eta * H_drive)

    def forward(self, x, H_drive: float = 0.0):
        if self.training:
            self.decay(H_drive)
        W_scaled = self.weight * self.gains.view(-1, 1) # output = (gain * W) · x + b
        return F.linear(x, W_scaled, self.bias)

# --- Network ---
class GainSlimResNet18(nn.Module):
    def __init__(self, input_size=(3,32,32), nf=20,
                 output_dim=10, g0=1.0, gamma=0.9,
                 eta=0.0):
        super().__init__()
        # 1) Backbone
        self.backbone = ResNet18feat(input_size, nf, global_pooling=True)
        feat_dim     = self.backbone.feature_size
        # 2) Head
        self.fc       = GainModLinear(feat_dim, output_dim,
                                      g0=g0, gamma=gamma,
                                      eta=eta)

    def forward(self, x, H_prev: float = 0.0):
        feats = self.backbone(x)           # (B, feat_dim)
        return self.fc(feats, H_prev)      # (B, output_dim)
    

## Continual evaluation
def continual_train(model, optimizer, criterion,
                    contexts, batch, ctx_iter, rho_eval,
                    train_ds, test_ds, device, mode):

    if isinstance(contexts[0][0], str):
        name_to_idx = {n: i for i, n in enumerate(train_ds.classes)}
        contexts = [[name_to_idx[c] for c in ctx] for ctx in contexts]

    train_targets = torch.tensor(train_ds.targets)
    test_targets  = torch.tensor(test_ds.targets)

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
        mask_train   = torch.isin(train_targets, torch.tensor(ctx))
        idx_train    = torch.nonzero(mask_train, as_tuple=False).squeeze()
        subset_train = Subset(train_ds, idx_train)
        sampler      = RandomSampler(subset_train,replacement=True,num_samples=ctx_iter * batch)
        train_loader = DataLoader(subset_train,batch_size=batch,sampler=sampler)
        train_iter   = iter(train_loader)

        # --------------------- TEST LOADER ---------------------------
        mask_test    = torch.isin(test_targets, torch.tensor([0,1]))
        idx_test     = torch.nonzero(mask_test, as_tuple=False).squeeze()
        subset_test  = Subset(test_ds, idx_test)
        #eval_sampler = RandomSampler(subset_test,replacement=False,num_samples=2000)
        #test_loader  = DataLoader(subset_test,batch_size=batch,sampler=eval_sampler)
        test_loader  = DataLoader(subset_test, batch_size=batch, shuffle=False) # all dataset each time
        eval_loaders[task_id] = test_loader

        # --------------------- TRAINING ------------------------------
        model.train()
        for _ in range(ctx_iter):
            inputs, targets = next(train_iter)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, H_prev)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # compute entropy 
            if mode == "gain-H":
                with torch.no_grad():
                    dist   = Categorical(logits=outputs)
                    H_prev = dist.entropy().mean().item()

            # --- train metrics ---
            hist["acc_train"].append((outputs.argmax(1) == targets).float().mean().item())
            hist["loss_train"].append(loss.item())

            # w_out/gain_out from final head
            hist["w_out"].append(model.fc.weight.data.abs().mean().item())
            hist["gain_out"].append(model.fc.gains.mean().item())

            # ---------------- CONTINUAL EVAL --------------------------
            if step % rho_eval == 0:
                model.eval()
                total, correct, loss_sum = 0, 0, 0.0
                with torch.no_grad():
                    for ev_loader in eval_loaders.values():
                        for x_t, y_t in ev_loader:
                            x_t, y_t = x_t.to(device), y_t.to(device)
                            logits = model(x_t, H_prev)
                            loss_ev = criterion(logits, y_t)
                            loss_sum += loss_ev.item() * y_t.size(0)
                            correct  += (logits.argmax(1) == y_t).sum().item()
                            total    += y_t.size(0)

                hist["acc_test"].append(100.0 * correct / total)
                hist["loss_test"].append(loss_sum / total)
                model.train()

            step += 1

    return hist

## Hyperparameters
batch = 256
lr_weights = 0.01
rho_eval = 1
ctx_iter = 400
contexts = [['airplane','automobile'], 
            ['airplane','automobile','bird','cat'], 
            ['airplane','automobile','bird','cat','deer','dog'], 
            ['airplane','automobile','bird','cat','deer','dog','frog','horse'], 
            ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']]
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
        hgainModel = GainSlimResNet18(output_dim=10,nf=20,g0=1.0, gamma=0.9, eta=0.2).to(device)
        hgainOptimizer = optim.SGD([ 
            {'params': [p for n, p in hgainModel.named_parameters() if 'gains' not in n and not n.startswith('fc.')],'lr': lr_weights,'momentum': 0.9},
            {'params': [p for n, p in hgainModel.named_parameters()if n == 'fc.weight'],'lr': lr_weights,'momentum': 0.0} 
        ])
        hgain_hist = continual_train(hgainModel, hgainOptimizer, criterion,
                                     contexts, batch, ctx_iter, rho_eval,
                                     train_ds, test_ds, device, mode="gain-H")

        # --- VANILLA ADAM ---
        print("VANILLA ADAM")
        adamModel = GainSlimResNet18(output_dim=10,nf=20,g0=1.0, gamma=0.0, eta=0.0).to(device)
        adamOptimizer = optim.Adam(
            [{'params':[p for n,p in adamModel.named_parameters() if 'gains' not in n and n!='fc.bias'],'lr':lr_weights}]
        )

        adam_hist = continual_train(adamModel, adamOptimizer, criterion,contexts, batch, ctx_iter, rho_eval, train_ds, test_ds, device, mode="no-gain")

        # --- MOMENTUM SGD ---
        print("MOMENTUM SGD")
        msgdModel = GainSlimResNet18(output_dim=10,nf=20,g0=1.0, gamma=0.0, eta=0.0).to(device)
        msgdOptimizer = optim.SGD(
            [{'params':[p for n,p in msgdModel.named_parameters() if 'gains' not in n and n!='fc.bias'],'lr':lr_weights}], 
            momentum=0.9
        )

        msgd_hist = continual_train(msgdModel, msgdOptimizer, criterion,contexts, batch, ctx_iter, rho_eval,train_ds, test_ds, device, mode="no-gain")

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