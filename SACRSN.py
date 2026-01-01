import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. Configuration (Refined)
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,
    "max_recursion_depth": 8,
    "act_threshold": 0.99,
    "ponder_penalty": 0.001,  # REDUCED by 10x to prevent lobotomy
    
    "n_symbols": 32,
    "n_concepts": 8,
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.1,  # REDUCED for stability
    
    "epochs": 1000,
    "learning_rate": 0.001,   # Reduced slightly
    "grad_clip": 0.5,         # Tighter clipping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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
data_tensor = torch.tensor([char_to_ix[c] for c in TEXT_DATA], dtype=torch.long).to(CONFIG["device"])

# ==========================================
# 3. Stabilized Complex Layers
# ==========================================
class ComplexLayerNorm(nn.Module):
    """
    CRITICAL FIX: Normalizes complex vectors to prevent explosion.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
    def forward(self, z):
        # Calculate mean and variance of magnitude
        mag = torch.abs(z) + CONFIG["eps"]
        mean = mag.mean(dim=-1, keepdim=True)
        var = mag.var(dim=-1, keepdim=True)
        
        # Normalize magnitude, preserve phase
        norm_mag = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        
        # Apply learnable affine transform
        norm_mag = norm_mag * self.scale + self.shift
        
        # Recombine with original phase
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
        # Xavier initialization for complex numbers
        nn.init.xavier_uniform_(self.fc_real.weight)
        nn.init.xavier_uniform_(self.fc_imag.weight)

    def forward(self, z):
        r, i = z.real, z.imag
        out_r = self.fc_real(r) - self.fc_imag(i)
        out_i = self.fc_real(i) + self.fc_imag(r)
        return torch.complex(out_r, out_i)

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim) # Added Norm
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)
        
        # FIX: Bias halting to NEGATIVE to encourage thinking at start
        # A bias of -2.0 means sigmoid(-2) ~= 0.12 initial prob of stopping
        nn.init.constant_(self.halt_linear.bias, -2.0) 

    def forward(self, z):
        z_new = self.linear(z)
        z_norm = self.norm(z_new) # Normalize before activation
        z_out = self.act(z_norm)
        
        z_flat = torch.cat([z_out.real, z_out.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        
        return z_out, halt_prob

# ==========================================
# 4. Graph-Augmented VQ (Bounded)
# ==========================================
class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.randn(n_symbols, latent_dim*2))
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        # L2 Distance
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        if prev_symbol_idx is not None:
            graph_prior = self.adjacency[prev_symbol_idx]
            # FIX: Clamp bias to avoid destroying the distance metric
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
# 5. Advanced CRSN Model
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

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1)
        
        if hidden is not None:
            z = 0.5 * z + 0.5 * hidden

        act_step = 0
        halting_probability = torch.zeros(batch_size, 1).to(z.device)
        remain = torch.ones(batch_size, 1).to(z.device)
        ponder_cost = 0
        
        z_weighted = torch.zeros_like(z) 
        
        current_sym = prev_sym
        vq_loss_total = 0
        
        for t in range(CONFIG["max_recursion_depth"]):
            act_step += 1
            
            z, p_halt = self.cell(z)
            z_sym, vq_loss, sym_idx = self.vq_layer(z, current_sym)
            current_sym = sym_idx
            
            # Weighted Skip Connection (ResNet style)
            z = 0.7*z + 0.3*z_sym
            
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            p = p_halt * still_running
            
            if t == CONFIG["max_recursion_depth"] - 1:
                p = remain
            
            z_weighted = z_weighted + (p * z)
            halting_probability = halting_probability + p
            remain = remain - p
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss

        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        return logits, z_weighted, current_sym, ponder_cost, vq_loss_total, act_step

# ==========================================
# 6. Training
# ==========================================
def train():
    model = AdvancedCRSN(vocab_size, CONFIG["embedding_dim"]).to(CONFIG["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    
    print(f"--- Training Stabilized Advanced CRSN ---")
    
    for epoch in range(CONFIG["epochs"]):
        hidden = None
        prev_sym = None
        
        total_loss = 0
        total_ponder = 0
        
        # Reset state more frequently to prevent long-term explosions
        window = 16 
        
        for i in range(0, len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            y = data_tensor[i+1].view(1)
            
            logits, hidden, sym_idx, ponder, vq_loss, steps = model(x, hidden, prev_sym)
            
            hidden = hidden.detach()
            prev_sym = sym_idx.detach()
            
            loss_pred = F.cross_entropy(logits, y)
            loss_ponder = CONFIG["ponder_penalty"] * ponder
            
            loss = loss_pred + loss_ponder + 0.1 * vq_loss
            
            # Check for NaN before backward
            if torch.isnan(loss):
                print(f"PANIC: NaN Loss at Epoch {epoch}, Step {i}")
                # Reset model state
                hidden = None
                continue

            opt.zero_grad()
            loss.backward()
            
            # Check for exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            
            opt.step()
            
            total_loss += loss.item()
            total_ponder += ponder.item()
            
            if i % window == 0:
                hidden = None
                prev_sym = None

        if epoch % 10 == 0:
            avg_ponder = total_ponder / len(data_tensor)
            avg_loss = total_loss / len(data_tensor)
            print(f"Ep {epoch:04d} | Loss: {avg_loss:.4f} | Avg Thinking Steps: {avg_ponder:.2f}")

    return model

# ==========================================
# 7. Visualization
# ==========================================
def visualize_brain(model):
    print("\n--- Visualizing Learned Symbolic Graph ---")
    adj = model.vq_layer.adjacency.detach().cpu().numpy()
    G = nx.DiGraph()
    n_sym = CONFIG["n_symbols"]
    for i in range(n_sym):
        for j in range(n_sym):
            weight = adj[i, j]
            if weight > 0.5:
                G.add_edge(f"S{i}", f"S{j}", weight=weight)
    
    if len(G.nodes) > 0:
        plt.figure(figsize=(8, 8))
        pos = nx.circular_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        nx.draw_networkx_nodes(G, pos, node_color='#a0cbe2', node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues)
        plt.show()
    else:
        print("Graph sparse.")

    model.eval()
    start_char = "T"
    x = torch.tensor([[char_to_ix[start_char]]], device=CONFIG["device"])
    hidden = None
    prev_sym = None
    
    print("\nGenerated: T", end="")
    for _ in range(100):
        with torch.no_grad():
            logits, hidden, prev_sym, _, _, steps = model(x, hidden, prev_sym)
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            print(ix_to_char[next_ix.item()], end="")
            x = next_ix
    print("\n")

if __name__ == "__main__":
    trained_model = train()
    visualize_brain(trained_model)
