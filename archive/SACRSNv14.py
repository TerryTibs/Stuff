# ============================================================
# SACRSNv8 — RESEARCH-READY MAXIMALIST
# ============================================================

import os, time, json, math, random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx

# ============================================================
# 0. DETERMINISM
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    "embedding_dim": 64,
    "n_symbols": 64,
    "max_recursion_depth": 8,
    "act_threshold": 0.999,
    "ponder_penalty": 1e-4,
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.5,
    "stack_size": 16,
    "epochs": 300,
    "lr": 1e-3,
    "grad_clip": 0.5,
    "grad_accum_steps": 4,
    "eps": 1e-6,
    "seed": SEED,
}

RUN_ID = f"run_{int(time.time())}"
RUN_DIR = os.path.join("runs", RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

# ============================================================
# 2. DATA
# ============================================================

TEXT = """True, without falsehood, certain and most true.
That which is above is like to that which is below."""

chars = sorted(set(TEXT))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
vocab_size = len(chars)

data = torch.tensor([stoi[c] for c in TEXT], device=DEVICE)

# ============================================================
# 3. COMPLEX PRIMITIVES
# ============================================================

class ComplexLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))

    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        m = mag.mean(-1, keepdim=True)
        v = mag.var(-1, keepdim=True)
        mag_n = (mag - m) / torch.sqrt(v + CONFIG["eps"])
        mag_n = mag_n * self.scale + self.shift
        return torch.polar(mag_n, torch.angle(z))

class ModReLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        return z * (F.relu(mag + self.b) / mag)

class ComplexLinear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.r = nn.Linear(d, d, bias=False)
        self.i = nn.Linear(d, d, bias=False)
        nn.init.xavier_uniform_(self.r.weight)
        nn.init.xavier_uniform_(self.i.weight)

    def forward(self, z):
        r,i = z.real, z.imag
        return torch.complex(self.r(r)-self.i(i), self.r(i)+self.i(r))

# ============================================================
# 4. DIFFERENTIABLE STACK
# ============================================================

class DifferentiableStack(nn.Module):
    def __init__(self, d, size):
        super().__init__()
        self.size = size
        self.mem = torch.zeros(size, d, device=DEVICE)
        self.ptr = torch.zeros(size, device=DEVICE)
        self.ptr[0] = 1.0

    def forward(self, value, control):
        push, pop, noop = control

        ptr_up = torch.roll(self.ptr, 1)
        ptr_down = torch.roll(self.ptr, -1)
        new_ptr = push * ptr_up + pop * ptr_down + noop * self.ptr
        new_ptr = new_ptr / (new_ptr.sum() + CONFIG["eps"])

        self.mem = push * torch.outer(ptr_up, value) + (1 - push) * self.mem
        self.ptr = new_ptr
        read = torch.sum(self.mem * self.ptr.unsqueeze(-1), dim=0)
        return read, self.ptr.clone()

# ============================================================
# 5. ACT CELL
# ============================================================

class ACTCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = ComplexLinear(d)
        self.norm = ComplexLayerNorm(d)
        self.act = ModReLU(d)
        self.halt = nn.Linear(d*2, 1)
        self.ctrl = nn.Linear(d*2, 3)
        nn.init.constant_(self.halt.bias, -2.0)

    def forward(self, z):
        z = self.act(self.norm(self.lin(z)))
        flat = torch.cat([z.real, z.imag], -1)
        p_halt = torch.sigmoid(self.halt(flat))
        stack_ctrl = F.softmax(self.ctrl(flat), dim=-1)
        return z, p_halt, stack_ctrl

# ============================================================
# 6. GRAPH VQ MEMORY
# ============================================================

class GraphVQ(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n, d*2))
        self.adj = nn.Parameter(torch.zeros(n, n))

    def forward(self, z, prev=None):
        zf = torch.cat([z.real, z.imag], -1)
        dists = (
            zf.pow(2).sum(-1, keepdim=True)
            + self.codebook.pow(2).sum(-1)
            - 2 * zf @ self.codebook.t()
        )
        if prev is not None:
            dists -= CONFIG["graph_bias_scale"] * torch.sigmoid(self.adj[prev])
        idx = torch.argmin(dists, -1)
        zq = self.codebook[idx]
        loss = F.mse_loss(zq, zf.detach()) + CONFIG["commitment_cost"] * F.mse_loss(zq.detach(), zf)
        zq = zf + (zq - zf).detach()
        h = zq.shape[-1]//2
        return torch.complex(zq[:,:h], zq[:,h:]), loss, idx

# ============================================================
# 7. SACRSN CORE
# ============================================================

class SACRSN(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["embedding_dim"]
        self.mag = nn.Embedding(vocab_size, d)
        self.phase = nn.Embedding(vocab_size, d)  # FIXED
        self.cell = ACTCell(d)
        self.vq = GraphVQ(d, CONFIG["n_symbols"])
        self.stack = DifferentiableStack(d, CONFIG["stack_size"])
        self.out = nn.Linear(d*2, vocab_size)

    def embed(self, x):
        return torch.polar(self.mag(x), self.phase(x))

    def forward(self, x, h=None, prev_sym=None, logs=None):
        z = self.embed(x).squeeze(1)
        if h is not None:
            z = 0.5*z + 0.5*h

        halt = torch.zeros(1,1,device=DEVICE)
        remain = torch.ones_like(halt)
        z_acc = torch.zeros_like(z)
        ponder = 0.0
        vq_loss = 0.0

        for t in range(CONFIG["max_recursion_depth"]):
            z, p, ctrl = self.cell(z)

            # Teacher forcing (stochastic)
            if self.training and random.random() < 0.3:
                ctrl = torch.tensor([[1.0,0.0,0.0]], device=DEVICE)

            stack_read, ptr = self.stack(z.real, ctrl[0])
            z = z + torch.complex(stack_read, torch.zeros_like(stack_read))

            zq, vql, sym = self.vq(z, prev_sym)
            z = 0.7*z + 0.3*zq

            running = (halt < CONFIG["act_threshold"]).float()
            pt = p * running
            if t == CONFIG["max_recursion_depth"]-1:
                pt = remain

            z_acc += pt*z
            halt += pt
            remain -= pt
            ponder += running.mean()
            vq_loss += vql
            prev_sym = sym

            if logs is not None:
                logs["symbols"].append(sym.item())
                logs["halt"].append(p.item())
                logs["stack_ptr"].append(ptr.cpu().numpy())

        logits = self.out(torch.cat([z_acc.real, z_acc.imag], -1))
        return logits, z_acc, prev_sym, ponder, vq_loss

# ============================================================
# 8. TRAINING + LOGGING
# ============================================================

import wandb
wandb.init(project="SACRSNv8")
wandb.config.update(CONFIG)

def get_curriculum_stage(epoch):
    if epoch < 50: return "simple"
    elif epoch < 150: return "medium"
    else: return "full"

def train():
    model = SACRSN().to(DEVICE)
    model = nn.DataParallel(model)  # Multi-GPU support
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    symbol_log = []
    halt_log = []
    stack_log = []
    loss_log = []

    grad_accum = CONFIG["grad_accum_steps"]

    for ep in range(CONFIG["epochs"]):
        h=None; prev=None
        total=0.0

        stage = get_curriculum_stage(ep)
        logs = {"symbols":[], "halt":[], "stack_ptr":[]}

        opt.zero_grad()
        for i in range(0, len(data)-1, grad_accum):
            batch_loss = 0.0
            for j in range(grad_accum):
                idx = i+j
                if idx >= len(data)-1: break
                x = data[idx].view(1,1)
                y = data[idx+1].view(1)
                logits,h,prev,ponder,vq = model(x,h,prev,logs)
                h=h.detach(); prev=prev.detach()
                loss = F.cross_entropy(logits,y) + CONFIG["ponder_penalty"]*ponder + 0.1*vq
                loss = loss / grad_accum
                loss.backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step(); opt.zero_grad()
            total += batch_loss

        loss_log.append(total)
        symbol_log.extend(logs["symbols"])
        halt_log.extend(logs["halt"])
        stack_log.extend(logs["stack_ptr"])

        wandb.log({
            "epoch": ep,
            "loss": total,
            "avg_ponder": ponder.item(),
            "symbol_entropy": np.unique(symbol_log, return_counts=True)[1].mean()
        })

        if ep%50==0:
            print(f"Epoch {ep:04d} | Loss {total:.4f}")

    np.save(os.path.join(RUN_DIR,"symbols.npy"), np.array(symbol_log))
    np.save(os.path.join(RUN_DIR,"halt.npy"), np.array(halt_log))
    np.save(os.path.join(RUN_DIR,"stack.npy"), np.array(stack_log))
    np.save(os.path.join(RUN_DIR,"loss.npy"), np.array(loss_log))
    torch.save(model.state_dict(), os.path.join(RUN_DIR,"model.pt"))
    return model

# ============================================================
# 9. GRAPH + RULE EXTRACTION
# ============================================================

def analyze(model):
    adj=torch.sigmoid(model.module.vq.adj).detach().cpu().numpy()
    G=nx.DiGraph()
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j]>0.4:
                G.add_edge(f"S{i}",f"S{j}",weight=float(adj[i,j]))
    nx.write_graphml(G, os.path.join(RUN_DIR,"graph.graphml"))
    plt.figure(figsize=(10,10))
    nx.draw(G,nx.circular_layout(G,seed=SEED),with_labels=True)
    plt.savefig(os.path.join(RUN_DIR,"graph.png"))
    plt.close()

# ============================================================
# 10. MAIN
# ============================================================

if __name__=="__main__":
    model=train()
    analyze(model)
    with open(os.path.join(RUN_DIR,"config.json"),"w") as f:
        json.dump(CONFIG,f,indent=2)
    print("SACRSNv14 complete:", RUN_DIR)
