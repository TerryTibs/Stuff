import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,
    "max_recursion_depth": 8, # Increased depth limit for ACT
    "act_threshold": 0.99,    # Stop thinking when 99% confident
    "ponder_penalty": 0.01,   # Cost per thinking step
    
    "n_symbols": 32,
    "n_concepts": 8,
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.5,  # How much the Graph memory influences choice
    
    "epochs": 200,
    "learning_rate": 0.002,
    "grad_clip": 1.0,
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
# 3. Advanced Complex Layers
# ==========================================
class ModReLU(nn.Module):
    """
    Phase-preserving activation.
    Squashes magnitude if it exceeds threshold, but keeps angle (phase) intact.
    """
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, z):
        # z shape: [..., dim] (complex)
        norm = torch.abs(z) + CONFIG["eps"]
        # ReLU acts on the magnitude + learnable bias
        scale = F.relu(norm + self.bias) / norm
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_real = nn.Linear(dim, dim, bias=False)
        self.fc_imag = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.fc_real.weight)
        nn.init.orthogonal_(self.fc_imag.weight)

    def forward(self, z):
        r, i = z.real, z.imag
        out_r = self.fc_real(r) - self.fc_imag(i)
        out_i = self.fc_real(i) + self.fc_imag(r)
        return torch.complex(out_r, out_i)

class AdaptiveRecursiveCell(nn.Module):
    """
    Now includes a 'Halting' neuron to decide when to stop thinking.
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.act = ModReLU(dim)
        # Halting gate: Complex -> Scalar Probability
        self.halt_linear = nn.Linear(dim * 2, 1) 

    def forward(self, z):
        z_new = self.linear(z)
        z_out = self.act(z_new)
        
        # Calculate Halting Probability for ACT
        z_flat = torch.cat([z_out.real, z_out.imag], dim=-1)
        halt_prob = torch.sigmoid(self.halt_linear(z_flat))
        
        return z_out, halt_prob

# ==========================================
# 4. Graph-Augmented VQ
# ==========================================
class GraphMemoryVQ(nn.Module):
    def __init__(self, latent_dim, n_symbols):
        super().__init__()
        self.n_symbols = n_symbols
        self.codebook = nn.Parameter(torch.randn(n_symbols, latent_dim*2))
        
        # ADJACENCY MATRIX: Learnable transitions between symbols
        # A[i, j] = Log-likelihood of transitioning from Symbol i to Symbol j
        self.adjacency = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def forward(self, z, prev_symbol_idx=None):
        z_flat = torch.cat([z.real, z.imag], dim=-1)
        
        # 1. Calculate Geometric Distance
        # (z - e)^2 = z^2 + e^2 - 2ze
        d = torch.sum(z_flat**2, dim=-1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=-1) - \
            2 * torch.matmul(z_flat, self.codebook.t())
        
        # 2. Apply Graph Bias (Predictive Coding)
        # If we visited 'prev_symbol' last time, reduce distance to likely neighbors
        if prev_symbol_idx is not None:
            # Get transition row for previous symbol
            # prev_symbol_idx shape: [Batch] -> [Batch, n_symbols]
            graph_prior = self.adjacency[prev_symbol_idx] # [Batch, N_Sym]
            
            # Subtracting prior from distance makes those symbols "closer"
            d = d - (CONFIG["graph_bias_scale"] * torch.sigmoid(graph_prior))

        # 3. Quantize
        min_indices = torch.argmin(d, dim=-1)
        z_q = F.embedding(min_indices, self.codebook)
        
        # Losses
        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)
        
        # Straight-Through
        z_q = z_flat + (z_q - z_flat).detach()
        
        # Re-construct complex
        half = z_q.shape[-1] // 2
        z_complex = torch.complex(z_q[..., :half], z_q[..., half:])
        
        return z_complex, loss_vq + loss_commit * CONFIG["commitment_cost"], min_indices

# ==========================================
# 5. The Advanced Model (ACT + Graph)
# ==========================================
class AdvancedCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        
        # Complex Embeddings
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        
        # Adaptive Reasoning Core
        self.cell = AdaptiveRecursiveCell(dim)
        
        # Graph Memory
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        
        # Readout
        self.decoder = nn.Linear(dim*2, vocab_size)

    def embed(self, idx):
        r = self.emb_mag(idx)
        t = self.emb_phase[idx]
        return torch.complex(r*torch.cos(t), r*torch.sin(t))

    def forward(self, input_ids, hidden=None, prev_sym=None):
        batch_size = input_ids.size(0)
        z = self.embed(input_ids).squeeze(1) # [B, Dim]
        
        if hidden is not None:
            z = 0.5 * z + 0.5 * hidden

        # ACT State variables
        act_step = 0
        halting_probability = torch.zeros(batch_size, 1).to(z.device) # Cumulative prob
        remain = torch.ones(batch_size, 1).to(z.device)               # Remaining budget
        ponder_cost = 0
        
        # The final 'thought' is a weighted average of all recursion steps
        z_weighted = torch.zeros_like(z) 
        
        # --- Adaptive Recursion Loop ---
        current_sym = prev_sym
        vq_loss_total = 0
        
        # We loop until max depth, but 'remain' will zero out contributions after halting
        for t in range(CONFIG["max_recursion_depth"]):
            act_step += 1
            
            # 1. Process
            z, p_halt = self.cell(z)
            
            # 2. Graph VQ (Symbolic Grounding)
            z_sym, vq_loss, sym_idx = self.vq_layer(z, current_sym)
            current_sym = sym_idx # Update context for next step
            
            # Residual mixing (Thought + Symbol)
            z = 0.6*z + 0.4*z_sym
            
            # 3. ACT Accumulation Logic
            # Mask for samples that haven't halted yet
            still_running = (halting_probability < CONFIG["act_threshold"]).float()
            
            # Effective probability for this step
            p = p_halt * still_running
            
            # Force halt at last step
            if t == CONFIG["max_recursion_depth"] - 1:
                p = remain
            
            # Accumulate weighted state
            z_weighted = z_weighted + (p * z)
            
            # Update counters
            halting_probability = halting_probability + p
            remain = remain - p
            
            ponder_cost += still_running.mean()
            vq_loss_total += vq_loss

        # Decode
        features = torch.cat([z_weighted.real, z_weighted.imag], dim=-1)
        logits = self.decoder(features)
        
        return logits, z_weighted, current_sym, ponder_cost, vq_loss_total, act_step

# ==========================================
# 6. Training with Graph & Ponder Losses
# ==========================================
def train():
    model = AdvancedCRSN(vocab_size, CONFIG["embedding_dim"]).to(CONFIG["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"--- Training Advanced CRSN (ACT + ModReLU + Graph) ---")
    
    loss_history = []
    
    for epoch in range(CONFIG["epochs"]):
        hidden = None
        prev_sym = None
        
        total_loss = 0
        total_ponder = 0
        
        # Truncated BPTT window
        window = 32
        
        for i in range(0, len(data_tensor) - 1):
            x = data_tensor[i].view(1, 1)
            y = data_tensor[i+1].view(1)
            
            # Forward
            logits, hidden, sym_idx, ponder, vq_loss, steps = model(x, hidden, prev_sym)
            
            # Update history (detach to prevent infinite graph)
            hidden = hidden.detach()
            prev_sym = sym_idx.detach()
            
            # Losses
            loss_pred = F.cross_entropy(logits, y)
            
            # Ponder Cost: Penalize thinking too long
            loss_ponder = CONFIG["ponder_penalty"] * ponder
            
            loss = loss_pred + loss_ponder + 0.1 * vq_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
            
            total_loss += loss.item()
            total_ponder += ponder.item()
            
            # Reset state occasionally to prevent drift
            if i % window == 0:
                hidden = None
                prev_sym = None

        if epoch % 20 == 0:
            avg_ponder = total_ponder / len(data_tensor)
            print(f"Ep {epoch:03d} | Loss: {total_loss:.2f} | Avg Thinking Steps: {avg_ponder:.2f}")

    return model

# ==========================================
# 7. Visualization
# ==========================================
def visualize_brain(model):
    print("\n--- Visualizing Learned Symbolic Graph ---")
    
    # 1. Extract Adjacency Matrix
    adj = model.vq_layer.adjacency.detach().cpu().numpy()
    
    # 2. Build Graph
    G = nx.DiGraph()
    n_sym = CONFIG["n_symbols"]
    
    # Add strong edges
    for i in range(n_sym):
        for j in range(n_sym):
            weight = adj[i, j]
            if weight > 0.5: # Threshold for visualization
                G.add_edge(f"S{i}", f"S{j}", weight=weight)
    
    # 3. Plot
    if len(G.nodes) > 0:
        plt.figure(figsize=(10, 8))
        pos = nx.circular_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        nx.draw_networkx_nodes(G, pos, node_color='#a0cbe2', node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, 
                               edge_cmap=plt.cm.Blues, width=2, arrowstyle='->', arrowsize=20)
        
        plt.title("Learned Symbolic Transition Graph (The 'Rules' it found)")
        plt.axis('off')
        plt.show()
    else:
        print("Graph weights too low to visualize yet. Train longer!")

    # 4. Generate Text
    model.eval()
    start_char = "T"
    x = torch.tensor([[char_to_ix[start_char]]], device=CONFIG["device"])
    hidden = None
    prev_sym = None
    output = start_char
    
    print("\nGenerated: ", end="")
    print(start_char, end="")
    
    for _ in range(150):
        with torch.no_grad():
            logits, hidden, prev_sym, _, _, steps = model(x, hidden, prev_sym)
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, 1)
            char = ix_to_char[next_ix.item()]
            print(char, end="")
            x = next_ix
            
    print("\n")

if __name__ == "__main__":
    trained_model = train()
    visualize_brain(trained_model)
