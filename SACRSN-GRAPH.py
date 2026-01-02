import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# ==========================================
# 1. Configuration (Must match training!)
# ==========================================
CONFIG = {
    "seq_len": 32,
    "embedding_dim": 64,
    "max_recursion_depth": 8,
    "act_threshold": 0.999,
    "ponder_penalty": 0.0001,  
    "n_symbols": 32,
    "n_concepts": 8,
    "commitment_cost": 0.25,
    "graph_bias_scale": 1.0, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eps": 1e-6
}

# Need text data to calculate vocab_size for the architecture
TEXT_DATA = """True, without falsehood, certain and most true. 
That which is above is like to that which is below, 
and that which is below is like to that which is above.
The father of all perfection in the whole world is here.
Its force or power is entire if it be converted into earth."""

chars = sorted(list(set(TEXT_DATA)))
vocab_size = len(chars)

# ==========================================
# 2. Architecture Definitions (REQUIRED TO LOAD)
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
    def forward(self, z):
        r, i = z.real, z.imag
        out_r = self.fc_real(r) - self.fc_imag(i)
        out_i = self.fc_real(i) + self.fc_imag(r)
        return torch.complex(out_r, out_i)

class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim) 
        self.act = ModReLU(dim)
        self.halt_linear = nn.Linear(dim * 2, 1)

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
        # Forward pass not needed for visualization, but class structure must exist
        pass

class AdvancedCRSN(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.emb_mag = nn.Embedding(vocab_size, dim)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, dim))
        self.cell = AdaptiveRecursiveCell(dim)
        self.vq_layer = GraphMemoryVQ(dim, CONFIG["n_symbols"])
        self.decoder = nn.Linear(dim*2, vocab_size)

# ==========================================
# 3. Helper Functions
# ==========================================
def load_model(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
        
    print(f"Loading from {path}...")
    # Load on CPU to ensure it works even if saved on GPU
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    # Initialize blank model
    model = AdvancedCRSN(vocab_size, CONFIG["embedding_dim"])
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Key mismatch! Are you using the exact same architecture code? \n{e}")
        return None

    model.eval()
    return model

def visualize_force(model):
    print("\n--- FORCE VISUALIZATION (Sigmoid Normalization) ---")
    
    # 1. Get Adjacency and Apply Sigmoid
    adj_logits = model.vq_layer.adjacency.detach().cpu()
    adj_probs = torch.sigmoid(adj_logits).numpy()
    
    G = nx.DiGraph()
    n_sym = CONFIG["n_symbols"]
    
    # 2. Build Graph (Threshold > 0.5)
    count = 0
    print("Analyzing connections...")
    for i in range(n_sym):
        for j in range(n_sym):
            weight = adj_probs[i, j]
            # Lowered threshold to 0.4 to ensure we see the faint logic
            if weight > 0.4: 
                G.add_edge(f"S{i}", f"S{j}", weight=weight)
                count += 1
    
    print(f"Found {count} connections (Threshold > 0.4 probability)")
    
    if count > 0:
        plt.figure(figsize=(12, 12))
        pos = nx.circular_layout(G)
        edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        # Nodes
        nx.draw_networkx_nodes(G, pos, node_color='#ff9999', node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        
        # Edges
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, 
                               edge_cmap=plt.cm.Blues, width=2, 
                               arrowstyle='->', arrowsize=20)
        
        plt.title("The Hidden Symbolic Graph (Learned Logic)", fontsize=15)
        plt.axis('off')
        plt.show()
        print("Graph displayed.")
    else:
        print("The Graph is flat. The weights are too low.")

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    FILENAME = "crsn_complex_model.pth"
    
    model = load_model(FILENAME)
    
    if model:
        visualize_force(model)
