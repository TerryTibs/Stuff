import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. Configuration (MAXIMALLY COMPLETE)
# ==========================================
CONFIG = {
    "embedding_dim": 64,

    # ACT / Reasoning
    "max_recursion_depth": 8,
    "act_threshold": 0.99,
    "ponder_penalty": 0.001,

    # Symbolic Memory
    "n_symbols": 32,
    "commitment_cost": 0.25,
    "graph_bias_scale": 0.1,

    # Training
    "epochs": 600,
    "learning_rate": 1e-3,
    "grad_clip": 0.5,
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

chars = sorted(set(TEXT_DATA))
vocab_size = len(chars)
char_to_ix = {c: i for i, c in enumerate(chars)}
ix_to_char = {i: c for i, c in enumerate(chars)}
data = torch.tensor([char_to_ix[c] for c in TEXT_DATA], device=CONFIG["device"])

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
        mag_norm = (mag - mean) / torch.sqrt(var + CONFIG["eps"])
        mag_norm = mag_norm * self.scale + self.shift
        phase = torch.angle(z)
        return torch.complex(mag_norm * torch.cos(phase),
                             mag_norm * torch.sin(phase))

class ModReLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        mag = torch.abs(z) + CONFIG["eps"]
        scale = F.relu(mag + self.bias) / mag
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.r = nn.Linear(dim, dim, bias=False)
        self.i = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.r.weight)
        nn.init.xavier_uniform_(self.i.weight)

    def forward(self, z):
        real = self.r(z.real) - self.i(z.imag)
        imag = self.r(z.imag) + self.i(z.real)
        return torch.complex(real, imag)

# ==========================================
# 4. Adaptive Recursive Cell (ACT)
# ==========================================
class AdaptiveRecursiveCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = ComplexLinear(dim)
        self.norm = ComplexLayerNorm(dim)
        self.act = ModReLU(dim)
        self.halt = nn.Linear(dim * 2, 1)
        nn.init.constant_(self.halt.bias, -2.0)

    def forward(self, z):
        z = self.lin(z)
        z = self.norm(z)
        z = self.act(z)
        halt_p = torch.sigmoid(
            self.halt(torch.cat([z.real, z.imag], dim=-1))
        )
        return z, halt_p

# ==========================================
# 5. Graph-Augmented VQ Memory
# ==========================================
class GraphMemoryVQ(nn.Module):
    def __init__(self, dim, n_symbols):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_symbols, dim * 2))
        self.adj = nn.Parameter(torch.zeros(n_symbols, n_symbols))

    def forward(self, z, prev_sym=None):
        zf = torch.cat([z.real, z.imag], dim=-1)
        d = (
            torch.sum(zf**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook**2, dim=-1)
            - 2 * zf @ self.codebook.t()
        )

        if prev_sym is not None:
            bias = CONFIG["graph_bias_scale"] * torch.sigmoid(self.adj[prev_sym])
            d = d - bias

        idx = torch.argmin(d, dim=-1)
        zq = F.embedding(idx, self.codebook)

        loss_vq = F.mse_loss(zq, zf.detach())
        loss_commit = F.mse_loss(zq.detach(), zf)

        zq = zf + (zq - zf).detach()
        half = zq.shape[-1] // 2
        zc = torch.complex(zq[..., :half], zq[..., half:])
        return zc, loss_vq + CONFIG["commitment_cost"] * loss_commit, idx

# ==========================================
# 6. FULL MODEL
# ==========================================
class AdvancedCRSN(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["embedding_dim"]
        self.emb_mag = nn.Embedding(vocab_size, d)
        self.emb_phase = nn.Parameter(torch.randn(vocab_size, d))
        self.cell = AdaptiveRecursiveCell(d)
        self.vq = GraphMemoryVQ(d, CONFIG["n_symbols"])
        self.decoder = nn.Linear(d * 2, vocab_size)

    def embed(self, x):
        r = self.emb_mag(x)
        t = self.emb_phase[x]
        return torch.complex(r * torch.cos(t), r * torch.sin(t))

    def forward(self, x, hidden=None, prev_sym=None):
        z = self.embed(x).squeeze(1)
        if hidden is not None:
            z = 0.5 * z + 0.5 * hidden

        halting_prob = torch.zeros(z.size(0), 1, device=z.device)
        remain = torch.ones_like(halting_prob)
        z_accum = torch.zeros_like(z)
        ponder = 0
        sym = prev_sym
        vq_loss_total = 0

        for t in range(CONFIG["max_recursion_depth"]):
            z, p = self.cell(z)
            z_sym, vq_loss, sym = self.vq(z, sym)
            z = 0.7 * z + 0.3 * z_sym

            still_running = (halting_prob < CONFIG["act_threshold"]).float()
            p_eff = p * still_running
            if t == CONFIG["max_recursion_depth"] - 1:
                p_eff = remain

            z_accum = z_accum + p_eff * z

            # 🔑 AUTOGRAD-SAFE FIX
            halting_prob = halting_prob + p_eff.detach()
            remain = remain - p_eff.detach()

            ponder += still_running.mean()
            vq_loss_total += vq_loss

        feats = torch.cat([z_accum.real, z_accum.imag], dim=-1)
        logits = self.decoder(feats)
        return logits, z_accum, sym, ponder, vq_loss_total

# ==========================================
# 7. Training
# ==========================================
def train():
    model = AdvancedCRSN().to(CONFIG["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    for ep in range(CONFIG["epochs"]):
        hidden = None
        sym = None
        total_loss = 0
        total_ponder = 0

        for i in range(len(data) - 1):
            x = data[i].view(1, 1)
            y = data[i + 1].view(1)

            logits, hidden, sym, ponder, vq_loss = model(x, hidden, sym)
            hidden = hidden.detach()
            sym = sym.detach()

            loss = (
                F.cross_entropy(logits, y)
                + CONFIG["ponder_penalty"] * ponder
                + 0.1 * vq_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()

            total_loss += loss.item()
            total_ponder += ponder.item()

            if i % 16 == 0:
                hidden = None
                sym = None

        if ep % 20 == 0:
            print(f"Ep {ep:04d} | Loss {total_loss:.3f} | Avg steps {total_ponder/len(data):.2f}")

    return model

# ==========================================
# 8. Visualization
# ==========================================
def visualize(model):
    print("\n--- Symbolic Graph ---")
    adj = model.vq.adj.detach().cpu().numpy()
    G = nx.DiGraph()

    for i in range(CONFIG["n_symbols"]):
        for j in range(CONFIG["n_symbols"]):
            if adj[i, j] > 0.5:
                G.add_edge(f"S{i}", f"S{j}")

    if G.nodes:
        plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True, node_size=600)
        plt.show()
    else:
        print("Graph still sparse.")

    print("\nGenerated:")
    x = torch.tensor([[char_to_ix["T"]]], device=CONFIG["device"])
    hidden = None
    sym = None
    out = "T"

    for _ in range(150):
        with torch.no_grad():
            logits, hidden, sym, _, _ = model(x, hidden, sym)
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs, 1)
            out += ix_to_char[x.item()]

    print(out)

# ==========================================
# 9. Run
# ==========================================
if __name__ == "__main__":
    m = train()
    visualize(m)
