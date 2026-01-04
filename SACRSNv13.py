import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import seaborn as sns
import os

# ==========================================
# 0. Reproducibility
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
torch.autograd.set_detect_anomaly(True)  # Detect inplace grad errors

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,
    "n_symbols": 64,
    "max_recursion_depth": 8,
    "act_threshold": 0.9999,
    "ponder_penalty": 0.001,
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.5,
    "symbol_consistency_weight": 0.01,
    "epochs": 300,
    "learning_rate": 0.001,
    "grad_clip": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eps": 1e-5
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
# 4. Adaptive Recursive Cell
# ==========================================
class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim*2, 1)
        nn.init.constant_(self.halt_linear.bias, -2.0)

    def forward(self, z):
        z_new = self.linear(z)
        z_norm = self.norm(z_new)
        z_out = self.act(z_norm)
        z_flat = torch.cat([z_out.real, z_out.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        return z_out, halt_prob

# ==========================================
# 5. Graph Memory Vector Quantizer
# ==========================================
class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.randn(n_symbols, latent_dim*2))
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - 2*torch.matmul(z_flat, self.codebook.t())

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
        return z_complex, loss_vq + loss_commit*CONFIG["commitment_cost"], min_indices

# ==========================================
# 6. Master Model
# ==========================================
class AdvancedCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)
        self.register_buffer("prev_sym_soft", torch.zeros(CONFIG["n_symbols"]))

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        if hidden is not None:
            z = 0.5*z + 0.5*hidden

        z_weighted = torch.zeros_like(z)
        halting_probability = torch.zeros(batch_size,1).to(z.device)
        remain = torch.ones(batch_size,1).to(z.device)
        ponder_cost = 0
        current_sym = prev_sym
        vq_loss_total = 0

        for t in range(CONFIG["max_recursion_depth"]):
            z, p_halt = self.cell(z)
            z_sym, vq_loss, sym_idx = self.vq_layer(z, current_sym)
            current_sym = sym_idx
            z = 0.7*z + 0.3*z_sym

            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            if t == CONFIG["max_recursion_depth"]-1:
                p = remain

            z_weighted = z_weighted + (p*z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        return logits, z_weighted, current_sym, ponder_cost, vq_loss_total, t+1

# ==========================================
# 7. Training Engine
# ==========================================
def train():
    model = AdvancedCRSN(vocab_size, CONFIG["embedding_dim"]).to(CONFIG["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"], eta_min=1e-5)

    for epoch in range(CONFIG["epochs"]):
        hidden, prev_sym = None, None
        total_loss, total_ponder = 0, 0
        window = 16
        entropy_weight = 0.01*(1 - epoch/CONFIG["epochs"])

        for i in range(len(data_tensor)-1):
            x = data_tensor[i].view(1,1)
            y = data_tensor[i+1].view(1)

            logits, hidden, sym_idx, ponder, vq_loss, steps = model(x, hidden, prev_sym)
            hidden, prev_sym = hidden.detach(), sym_idx.detach()

            # Losses
            loss_pred = F.cross_entropy(logits, y)
            loss_ponder = CONFIG["ponder_penalty"]*ponder
            probs = F.softmax(logits, dim=-1)
            loss_entropy = -entropy_weight * (-(probs*torch.log(probs+CONFIG["eps"])).sum())

            adj_sig = torch.sigmoid(model.vq_layer.adjacency)
            loss_static_consistency = CONFIG["symbol_consistency_weight"] * \
                (-(adj_sig*torch.log(adj_sig+CONFIG["eps"])).sum(dim=-1).mean())

            current_sym_onehot = F.one_hot(sym_idx, CONFIG["n_symbols"]).float().view(-1)
            loss_temporal = CONFIG["symbol_consistency_weight"]*F.mse_loss(current_sym_onehot, model.prev_sym_soft.detach())
            model.prev_sym_soft.mul_(0.9).add_(current_sym_onehot*0.1)

            loss = loss_pred + loss_ponder + 0.1*vq_loss + loss_entropy + loss_static_consistency + loss_temporal

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()

            total_loss += loss.item()
            total_ponder += ponder.item()
            if i%window==0: hidden, prev_sym = None, None

        scheduler.step()
        if epoch%50==0:
            print(f"Epoch {epoch} | Loss {total_loss/len(data_tensor):.4f} | Ponder {total_ponder/len(data_tensor):.2f}")

    return model

# ==========================================
# 8. Visualization
# ==========================================
def visualize_brain(model):
    adj_probs = torch.sigmoid(model.vq_layer.adjacency).detach().cpu().numpy()
    G = nx.DiGraph()
    n_sym = CONFIG["n_symbols"]
    for i in range(n_sym):
        for j in range(n_sym):
            if adj_probs[i,j]>0.4:
                G.add_edge(f"S{i}", f"S{j}", weight=adj_probs[i,j])
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items()) if G.edges else ([],[])
    nx.draw_networkx_nodes(G,pos,node_color="#a0cbe2",node_size=600)
    nx.draw_networkx_labels(G,pos,font_size=8)
    nx.draw_networkx_edges(G,pos,edgelist=edges,edge_color=weights,edge_cmap=plt.cm.Blues,width=2,arrowstyle='->',arrowsize=15)
    plt.title("Learned Symbolic Graph")
    plt.axis('off')
    plt.show()

# ==========================================
# 9. Logic Rule Extraction
# ==========================================
def extract_logic_rules(model, data_tensor):
    rule_book = defaultdict(list)
    hidden, prev_sym = None, None
    for i in range(len(data_tensor)-1):
        x = data_tensor[i].view(1,1)
        logits, hidden, sym_idx, ponder, _, _ = model(x, hidden, prev_sym)
        if prev_sym is not None:
            rule_book[(prev_sym.item(), sym_idx.item())].append(ponder.item())
        prev_sym = sym_idx
    print(f"\n{'FROM':<6} | {'TO':<6} | {'COUNT':<6} | {'AVG_STEPS':<10}")
    print("-"*40)
    for (src,dst), ponders in sorted(rule_book.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"S_{src:<4} -> S_{dst:<4} | {len(ponders):<6} | {np.mean(ponders):.2f}")

# ==========================================
# 10. Main
# ==========================================
if __name__=="__main__":
    FILENAME = "crsn_ultimate_final.pth"
    trained_model = train()
    torch.save({'model_state_dict': trained_model.state_dict(),'config': CONFIG}, FILENAME)
    print(f"Model saved to {FILENAME}")
    visualize_brain(trained_model)
    extract_logic_rules(trained_model, data_tensor)
