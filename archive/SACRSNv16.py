# ============================================================
# UBER-SACRSN: SCIENTIFIC EDITION
# Includes: Graph Topology, Stack MRI, Phase Plots, & Logic Extraction
# ============================================================

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# ==========================================
# 0. Determinism
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,      
    "n_symbols": 64,          
    
    # Reasoning & Memory
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.0001,
    "use_stack": True,
    "stack_size": 16,
    
    # Topology
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.5, 
    "symbol_consistency_weight": 0.01,
    
    # Training
    "epochs": 3000,
    "learning_rate": 0.001,
    "grad_clip": 0.5,
    "eps": 1e-6
}

# ==========================================
# 2. Data
# ==========================================
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(DEVICE)

# ==========================================
# 3. Complex Primitives
# ==========================================
class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        norm_mag = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        norm_mag = norm_mag * self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(norm_mag * torch.cos(phase), norm_mag * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, z):
        norm = torch.abs(z) + CONFIG["eps"]
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.fc_real.weight)
        nn.init.xavier_uniform_(self.fc_imag.weight)
    def forward(self, z):
        r, i = z.real, z.imag
        out_r = self.fc_real(r) - self.fc_imag(i)
        out_i = self.fc_real(i) + self.fc_imag(r)
        return torch.complex(out_r, out_i)

# ==========================================
# 4. Modules: Stack, Cell, VQ
# ==========================================
class DifferentiableStack(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.dim = dim
        self.size = size
    def forward(self, z, memory, ptr, control):
        push, pop, noop = control[:, 0].view(-1,1), control[:, 1].view(-1,1), control[:, 2].view(-1,1)
        ptr_up = torch.roll(ptr, 1, dims=1)
        ptr_down = torch.roll(ptr, -1, dims=1)
        new_ptr = (push * ptr_up) + (pop * ptr_down) + (noop * ptr)
        new_ptr = new_ptr / (new_ptr.sum(dim=1, keepdim=True) + CONFIG["eps"])
        
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        write_mask = push * ptr_up
        write_val = write_mask.unsqueeze(2) * z_flat.unsqueeze(1)
        retain_mask = 1.0 - write_mask.unsqueeze(2)
        new_memory = write_val + (memory * retain_mask)
        
        read_mask = new_ptr.unsqueeze(2)
        read_flat = torch.sum(new_memory * read_mask, dim=1)
        read_z = torch.complex(read_flat[:, :self.dim], read_flat[:, self.dim:])
        return read_z, new_memory, new_ptr

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim) 
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        self.stack_ctrl = nn.Linear(dim * 2, 3)
        nn.init.constant_(self.halt_linear.bias, -2.0) 
    def forward(self, z):
        z_proc = self.act(self.norm(self.linear(z)))
        z_flat = torch.cat([z_proc.real, z_proc.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        stack_probs = F.softmax(self.stack_ctrl(z_flat), dim=-1)
        return z_proc, halt_prob, stack_probs

class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.randn(n_symbols, latent_dim*2))
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))
    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        if prev_symbol_idx is not None:
            graph_prior = self.adjacency[prev_symbol_idx]
            bias = CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior)
            d = d - bias
        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        z_q = z_flat + (z_q - z_flat).detach()
        z_complex = torch.complex(z_q[..., :z.shape[-1]], z_q[..., z.shape[-1]:])
        
        # Calculate Perplexity (Usage)
        encodings = F.one_hot(min_indices, self.n_symbols).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices, perplexity

# ==========================================
# 5. Master Model
# ==========================================
class UberCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        if CONFIG["use_stack"]:
            self.stack = DifferentiableStack(dim, CONFIG["stack_size"])
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        if hidden is not None: z = 0.5 * z + 0.5 * hidden

        act_step = 0
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = 0
        
        # Stack initialization
        stack_mem = torch.zeros(batch_size, CONFIG["stack_size"], self.dim*2, device=z.device)
        stack_ptr = torch.zeros(batch_size, CONFIG["stack_size"], device=z.device)
        stack_ptr[:, 0] = 1.0
        stack_history = [] # For visualization

        z_weighted = torch.zeros_like(z) 
        current_sym = prev_sym
        vq_loss_total = 0
        perplexity_total = 0
        
        for t in range(CONFIG["max_recursion_depth"]):
            act_step += 1
            z_proc, p_halt, stack_ctrl = self.cell(z)
            
            if CONFIG["use_stack"]:
                stack_read, stack_mem, stack_ptr = self.stack(z_proc, stack_mem, stack_ptr, stack_ctrl)
                z_combined = z_proc + stack_read
                # Log stack depth (argmax of pointer)
                stack_history.append(torch.argmax(stack_ptr, dim=1).float().mean().item())
            else:
                z_combined = z_proc
                stack_history.append(0)

            z_sym, vq_loss, sym_idx, perplexity = self.vq_layer(z_combined, current_sym)
            current_sym = sym_idx
            
            z = 0.7 * z_combined + 0.3 * z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"] - 1: p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss
            perplexity_total += perplexity

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        return logits, z_weighted, current_sym, ponder_cost, vq_loss_total, perplexity_total/act_step, stack_history

# ==========================================
# 6. Training
# ==========================================
def train():
    model = UberCRSN(vocab_size, CONFIG["embedding_dim"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    
    print(f"--- Training UBER-SACRSN Scientific Edition ---")
    
    try:
        for epoch in range(CONFIG["epochs"]):
            hidden = None
            prev_sym = None
            total_loss = 0
            total_ponder = 0
            total_ppx = 0
            window = 16 
            entropy_weight = 0.01 * (1 - epoch / CONFIG["epochs"])
            
            for i in range(0, len(data_tensor) - 1):
                x = data_tensor[i].view(1, 1)
                y = data_tensor[i+1].view(1)
                
                # Forward Pass (Note the extra returns)
                logits, hidden, sym_idx, ponder, vq_loss, ppx, _ = model(x, hidden, prev_sym)
                hidden = hidden.detach()
                prev_sym = sym_idx.detach()
                
                loss_pred = F.cross_entropy(logits, y)
                loss_ponder = CONFIG["ponder_penalty"] * ponder
                
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                loss_entropy = -entropy_weight * (-(probs * log_probs).sum())
                
                adj_sig = torch.sigmoid(model.vq_layer.adjacency)
                row_entropy = -(adj_sig * torch.log(adj_sig + CONFIG["eps"])).sum(dim=-1).mean()
                loss_static = CONFIG["symbol_consistency_weight"] * row_entropy
                
                curr_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float()
                if curr_onehot.dim() > 1: curr_onehot = curr_onehot.view(-1)
                loss_temporal = CONFIG["symbol_consistency_weight"] * F.mse_loss(curr_onehot, model.prev_sym_soft.detach())
                model.prev_sym_soft.mul_(0.9).add_(curr_onehot * 0.1)
                
                loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_static + loss_temporal
                
                if torch.isnan(loss):
                    print(f"PANIC: NaN Loss at Ep {epoch}")
                    hidden = None; continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                
                total_loss += loss.item()
                total_ponder += ponder.item()
                total_ppx += ppx.item()
                if i % window == 0: hidden = None; prev_sym = None

            scheduler.step()

            if epoch % 50 == 0:
                avg_loss = total_loss / len(data_tensor)
                avg_ponder = total_ponder / len(data_tensor)
                avg_ppx = total_ppx / len(data_tensor)
                lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Steps: {avg_ponder:.2f} | Usage(PPX): {avg_ppx:.1f} | LR: {lr:.6f}")
                
                if avg_loss < 0.01:
                    print("\n--- PERFECT CONVERGENCE ---")
                    return model

    except KeyboardInterrupt:
        print("\nInterrupted.")
    
    return model

# ==========================================
# 7. Visualization Suite
# ==========================================
def visualize_all(model):
    print("\n--- Generating Diagnostics ---")
    model.eval()
    
    # 1. Graph Topology
    adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    G = nx.DiGraph()
    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            if adj_probs[i, j] > 0.4: G.add_edge(f"S{i}", f"S{j}", weight=adj_probs[i, j])
    
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    plt.figure(figsize=(10, 10))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='#a0cbe2', node_size=600, font_size=8)
    plt.title(f"Logical Topology (Active Nodes: {len(G.nodes)})")
    plt.show()

    # 2. Run Inference for Stack & Phase Plot
    hidden, prev_sym = None, None
    x = torch.tensor([[char_to_ix["T"]]], device=DEVICE)
    
    stack_depths = []
    complex_points = []
    generated_text = "T"
    
    print("Generating...")
    for _ in range(200):
        with torch.no_grad():
            # Get stack history from forward pass
            logits, hidden, prev_sym, _, _, _, s_hist = model(x, hidden, prev_sym)
            
            # Record Stack Depth (Average depth during the recursion steps)
            stack_depths.append(np.mean(s_hist))
            
            # Record Complex State (Latent Space)
            z = hidden.cpu().squeeze()
            complex_points.append(z)
            
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            char = ix_to_char[next_ix.item()]
            generated_text += char
            x = next_ix

    print(f"Output: {generated_text}\n")

    # 3. Stack MRI (Depth vs Time)
    plt.figure(figsize=(12, 4))
    plt.plot(stack_depths, label="Stack Depth", color='purple')
    plt.fill_between(range(len(stack_depths)), stack_depths, color='purple', alpha=0.1)
    plt.title("Stack MRI: Memory Depth over Time")
    plt.xlabel("Time Step (Character)")
    plt.ylabel("Stack Pointer Height")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4. Phase Constellation (Complex Space)
    # Extract real and imag components
    # We only take the first dimension of the complex vector for 2D plot
    reals = [c.real[0].item() for c in complex_points]
    imags = [c.imag[0].item() for c in complex_points]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(reals, imags, c=range(len(reals)), cmap='plasma', alpha=0.6)
    plt.colorbar(label="Time Step")
    plt.title("Phase Constellation (Latent Trajectory)")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# ==========================================
# 8. Main
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_scientific_model.pth"
    
    trained_model = train()
    
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': CONFIG,
    }, FILENAME)
    
    visualize_all(trained_model)
    
