import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import seaborn as sns

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,
    "n_symbols": 64,
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.0001,
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.5,
    "symbol_consistency_weight": 0.01,
    "epochs": 300,
    "learning_rate": 0.001,
    "grad_clip": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eps": 1e-6
}

# ==========================================
# 2. Data Preparation
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
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(CONFIG["device"])

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
# 4. Adaptive Recursive Cell & VQ
# ==========================================
class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim) 
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        nn.init.constant_(self.halt_linear.bias, -2.0) 

    def forward(self, z):
        z_new = self.linear(z)
        z_norm = self.norm(z_new)
        z_out = self.act(z_norm)
        z_flat = torch.cat([z_out.real, z_out.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        return z_out, halt_prob

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
        half = z_q.shape[-1] // 2
        z_complex = torch.complex(z_q[..., :half], z_q[..., half:])
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices

# ==========================================
# 5. Advanced CRSN v2
# ==========================================
class AdvancedCRSN_v2(nn.Module):
    def __init__(self, vocab_size, dim, tau=0.2):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        r = r / (r.norm(dim=-1, keepdim=True) + CONFIG["eps"])
        t = self.emb_phase[idx]
        return torch.complex(r * torch.cos(t), r * torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        if hidden is not None: z = 0.5*z + 0.5*hidden
        halting_probability = torch.zeros(batch_size,1,device=z.device)
        remain = torch.ones(batch_size,1,device=z.device)
        ponder_cost = torch.zeros(batch_size,1,device=z.device)
        z_weighted = torch.zeros_like(z)
        current_sym = prev_sym
        vq_loss_total = 0
        act_step = 0

        while (halting_probability < CONFIG["act_threshold"]).any() and act_step < CONFIG["max_recursion_depth"]:
            act_step += 1
            z, p_halt = self.cell(z)
            z_q, vq_loss, sym_idx = self.vq_layer(z, current_sym)

            # Temperature scaling & soft VQ
            z_flat = torch.cat([z.real,z.imag],dim=-1)
            dist = -torch.cdist(z_flat.unsqueeze(0), self.vq_layer.codebook.unsqueeze(0))/self.tau
            prob = F.softmax(dist.squeeze(0), dim=-1)
            z_q_smooth = torch.matmul(prob, self.vq_layer.codebook)
            half = z_q_smooth.shape[-1]//2
            z_q = torch.complex(z_q_smooth[...,:half], z_q_smooth[...,half:])
            current_sym = sym_idx

            # Weighted skip connection
            z = 0.7*z + 0.3*z_q

            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = torch.clamp(p_halt * still_running, 1e-6, 1.0)
            if act_step == CONFIG["max_recursion_depth"]: p = remain
            z_weighted = z_weighted + p*z
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost = ponder_cost + still_running.mean()
            vq_loss_total += vq_loss

        features = torch.cat([z_weighted.real,z_weighted.imag],dim=-1)
        logits = self.decoder(features)
        return logits, z_weighted, current_sym, ponder_cost.mean(), vq_loss_total, act_step

# ==========================================
# 6. Training Engine (Fixed)
# ==========================================
def train_improved_v2():
    model = AdvancedCRSN_v2(vocab_size, CONFIG["embedding_dim"]).to(CONFIG["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)
    seq_len = 4

    for epoch in range(CONFIG["epochs"]):
        total_loss, total_ponder = 0.0,0.0
        hidden, prev_sym = None,None
        entropy_weight = 0.01*(1-epoch/CONFIG["epochs"])
        indices = torch.randperm(len(data_tensor)-seq_len)

        for idx in indices:
            x_seq = data_tensor[idx:idx+seq_len].view(seq_len,1).to(CONFIG["device"])
            y_seq = data_tensor[idx+1:idx+seq_len+1].view(seq_len).to(CONFIG["device"])
            batch_loss, batch_ponder = 0.0,0.0

            for t in range(seq_len):
                x,y = x_seq[t].view(1,1), y_seq[t].view(1)
                logits, hidden, sym_idx, ponder, vq_loss, steps = model(x, hidden, prev_sym)
                hidden = hidden.detach()
                prev_sym = sym_idx.detach()

                # Loss components
                loss_pred = F.cross_entropy(logits,y)
                loss_ponder = CONFIG["ponder_penalty"]*ponder
                loss_vq = 0.1*vq_loss
                probs = F.softmax(logits,dim=-1)
                log_probs = F.log_softmax(logits,dim=-1)
                loss_entropy = -entropy_weight*(-(probs*log_probs).sum())
                adj_sig = torch.sigmoid(model.vq_layer.adjacency)
                row_entropy = -(adj_sig*torch.log(adj_sig+CONFIG["eps"])).sum(dim=-1).mean()
                loss_static_consistency = CONFIG["symbol_consistency_weight"]*row_entropy

                # Out-of-place temporal buffer update
                current_sym_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float().view(-1)
                loss_temporal = CONFIG["symbol_consistency_weight"]*F.mse_loss(current_sym_onehot, model.prev_sym_soft.detach())
                model.prev_sym_soft = model.prev_sym_soft * 0.9 + current_sym_onehot*0.1

                loss = loss_pred + loss_ponder + loss_vq + loss_entropy + loss_static_consistency + loss_temporal
                loss = torch.clamp(loss,0.0,1e3)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                opt.step()
                batch_loss += loss.item()
                batch_ponder += ponder.item()

            total_loss += batch_loss/seq_len
            total_ponder += batch_ponder/seq_len

        scheduler.step()
        if epoch%20==0:
            print(f"Epoch {epoch:04d} | Loss: {total_loss/len(indices):.4f} | Ponder: {total_ponder/len(indices):.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        if total_loss/len(indices)<0.01:
            print("--- PERFECT CONVERGENCE ---")
            break
    return model

# ==========================================
# 7. Visualization & Logic Extraction
# ==========================================
def visualize_brain_v2(model, data_tensor, start_char="T", gen_len=200):
    model.eval()
    adj_logits = model.vq_layer.adjacency.detach().cpu()
    adj_probs = torch.sigmoid(adj_logits).numpy()
    plt.figure(figsize=(8,6))
    sns.heatmap(adj_probs,cmap="viridis")
    plt.title("Soft Adjacency Heatmap of Learned Symbol Graph")
    plt.xlabel("Destination Symbol")
    plt.ylabel("Source Symbol")
    plt.show()

    G = nx.DiGraph()
    n_sym = CONFIG["n_symbols"]
    for i in range(n_sym):
        for j in range(n_sym):
            weight = adj_probs[i,j]
            if weight > 0.2: G.add_edge(f"S{i}",f"S{j}",weight=weight)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    if len(G.nodes)>0:
        plt.figure(figsize=(10,10))
        pos = nx.circular_layout(G)
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        nx.draw_networkx_nodes(G,pos,node_color="#a0cbe2",node_size=600)
        nx.draw_networkx_labels(G,pos,font_size=8)
        nx.draw_networkx_edges(G,pos,edgelist=edges,edge_color=weights,edge_cmap=plt.cm.Blues,width=2,arrowstyle='->',arrowsize=15)
        plt.title("Learned Symbolic Graph Topology")
        plt.axis('off')
        plt.show()
    else:
        print("Graph is sparse.")

    hidden, prev_sym = None,None
    x = torch.tensor([[char_to_ix[start_char]]], device=CONFIG["device"])
    traj_real, traj_imag = [],[]
    print("\nGenerated sequence:")
    print(start_char,end="")
    for _ in range(gen_len):
        with torch.no_grad():
            logits, hidden, prev_sym, _, _, _ = model(x,hidden,prev_sym)
            probs = F.softmax(logits,dim=-1)
            next_ix = torch.multinomial(probs,1)
            print(ix_to_char[next_ix.item()],end="")
            x = next_ix
            z = model.embed(x)
            traj_real.append(z.real.cpu().numpy().flatten()[0])
            traj_imag.append(z.imag.cpu().numpy().flatten()[0])
    print("\n")
    plt.figure(figsize=(8,6))
    plt.plot(traj_real,traj_imag,marker="o",markersize=3,linestyle='-')
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Symbol Trajectory in Complex Plane")
    plt.grid(True)
    plt.show()

def extract_logic_rules_v2(model, data_tensor):
    model.eval()
    rule_book = defaultdict(list)
    hidden, prev_sym = None, None
    with torch.no_grad():
        for i in range(len(data_tensor)-1):
            x = data_tensor[i].view(1,1)
            logits, hidden, sym_idx, ponder, _, _ = model(x, hidden, prev_sym)
            if prev_sym is not None:
                src,dst = prev_sym.item(), sym_idx.item()
                rule_book[(src,dst)].append(ponder.item())
            prev_sym = sym_idx
    print("\n--- Extracted Symbolic Logic Rules ---")
    print(f"{'FROM':<6} | {'TO':<6} | {'COUNT':<6} | {'AVG PONDER':<10}")
    print("-"*45)
    sorted_rules = sorted(rule_book.items(), key=lambda x: len(x[1]), reverse=True)
    for (src,dst),ponders in sorted_rules:
        count = len(ponders)
        avg_ponder = sum(ponders)/count
        if count>1: print(f"S_{src:<4} -> S_{dst:<4} | {count:<6} | {avg_ponder:<10.2f}")

# ==========================================
# 8. Main Execution
# ==========================================
if __name__=="__main__":
    FILENAME = "crsn_nextgen_final.pth"
    trained_model = train_improved_v2()
    print(f"\n--- Saving Model to {FILENAME} ---")
    torch.save({'model_state_dict': trained_model.state_dict(),'config': CONFIG}, FILENAME)
    print("Saved.")
    visualize_brain_v2(trained_model,data_tensor)
    extract_logic_rules_v2(trained_model,data_tensor)
