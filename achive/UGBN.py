import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os
from collections import deque, defaultdict

# ============================================================
# Configuration & Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Hyperparameters
GRID_SIZE = 12
NUM_AGENTS = 4
PROTOTYPE_DIM = 16
COMM_DIM = 8
HIDDEN_DIM = 64
HORIZON = 5
MCTS_SIMS = 20
LR = 0.002

# ============================================================
# 1. Multi-Agent Snake Environment
# ============================================================
class MultiSnakeEnv:
    def __init__(self, num_agents=4, grid_size=10):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.snakes = []
        self.food = None
        self.dead = [] # Track dead agents
        self.reset()

    def reset(self):
        self.snakes = []
        self.dead = [False] * self.num_agents
        # Spawn snakes at random positions
        for _ in range(self.num_agents):
            pos = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
            self.snakes.append([pos])
        self.food = self._spawn_food()
        return self._get_state()

    def _spawn_food(self):
        attempts = 0
        while attempts < 100:
            f = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            occupied = False
            for s in self.snakes:
                if f in s: occupied = True
            if not occupied:
                return f
            attempts += 1
        return (0, 0) # Fallback

    def _get_state(self):
        states = []
        for i, s in enumerate(self.snakes):
            if self.dead[i]:
                # Dead agents get zero vectors
                states.append((np.zeros(6, dtype=np.float32), {"NEAR_WALL": False, "FOOD_VISIBLE": False}))
                continue

            hx, hy = s[0]
            fx, fy = self.food
            
            # Vector input: [NormX, NormY, FoodX, FoodY, RelFoodX, RelFoodY]
            vec = np.array([
                hx / self.grid_size,
                hy / self.grid_size,
                fx / self.grid_size,
                fy / self.grid_size,
                (fx - hx) / self.grid_size,
                (fy - hy) / self.grid_size
            ], dtype=np.float32)

            # Symbolic raw input (Hard-coded perception sensors)
            raw = {
                "NEAR_WALL": hx <= 0 or hx >= self.grid_size - 1 or hy <= 0 or hy >= self.grid_size - 1,
                "FOOD_VISIBLE": True 
            }
            states.append((vec, raw))
        return states

    def step(self, actions):
        rewards = []
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
        
        # Calculate new heads
        new_heads = []
        for i, action in enumerate(actions):
            if self.dead[i]:
                new_heads.append(None)
                rewards.append(0.0)
                continue
            
            hx, hy = self.snakes[i][0]
            dx, dy = moves[action]
            new_heads.append((hx + dx, hy + dy))

        # Check collisions
        for i, nh in enumerate(new_heads):
            if self.dead[i]: continue

            reward = -0.01 # Step penalty
            is_dead = False
            
            # Wall collision
            if not (0 <= nh[0] < self.grid_size and 0 <= nh[1] < self.grid_size):
                is_dead = True
            
            # Self collision
            if nh in self.snakes[i]:
                is_dead = True
                
            # Collision with others
            for j, other in enumerate(self.snakes):
                if nh in other: # Head-to-body or Head-to-Head
                    is_dead = True
            
            if is_dead:
                reward = -1.0
                self.dead[i] = True
            else:
                # Move
                self.snakes[i].insert(0, nh)
                if nh == self.food:
                    reward = 1.0
                    self.food = self._spawn_food()
                else:
                    self.snakes[i].pop()
            
            rewards.append(reward)

        return self._get_state(), rewards, self.dead

    def render(self):
        # ASCII Render
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Food
        grid[self.food[1]][self.food[0]] = 'F'
        
        # Snakes
        for i, s in enumerate(self.snakes):
            if self.dead[i]: continue
            for j, (x, y) in enumerate(s):
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    grid[y][x] = str(i) if j == 0 else '#'
        
        print("-" * (self.grid_size + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.grid_size + 2))

# ============================================================
# 2. Neuro-Symbolic Components
# ============================================================
class Symbol:
    def __init__(self, name, dim, persistence=0.7):
        self.name = name
        # Prototype is a learnable vector representing the concept
        self.prototype = torch.randn(dim, device=device) * 0.1
        self.activation = 0.0
        self.persistence = persistence

class SymbolMemory:
    def __init__(self):
        self.working = deque(maxlen=20)
        self.long_term = defaultdict(float)

    def update(self, symbols):
        snapshot = {k: v.activation for k, v in symbols.items()}
        self.working.append(snapshot)
        # Consolidate to LTM
        for k, v in snapshot.items():
            self.long_term[k] = 0.995 * self.long_term[k] + 0.005 * v

    def novelty(self):
        if len(self.working) < 2: return 0.0
        # Simple L1 distance between last two cognitive states
        a = self.working[-1]
        b = self.working[-2]
        return sum(abs(a[k] - b.get(k, 0)) for k in a)

class HierarchicalResonance(nn.Module):
    def __init__(self, input_dim, prototype_dim=16, comm_dim=8, lr=0.1):
        super().__init__()
        self.lr = lr
        self.prototype_dim = prototype_dim
        
        # Definition of the Agent's Ontology
        self.low = {"FOOD": Symbol("FOOD", prototype_dim), "WALL": Symbol("WALL", prototype_dim)}
        self.mid = {"GOAL": Symbol("GOAL", prototype_dim), "THREAT": Symbol("THREAT", prototype_dim)}
        self.high = {"SURVIVAL": Symbol("SURVIVAL", prototype_dim)}

        # Encoder: Merges physical sensors and social communication
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + comm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, prototype_dim),
            nn.Tanh() # Latent space bounded [-1, 1]
        ).to(device)
        
        self.comm_out = nn.Sequential(nn.Linear(prototype_dim, comm_dim), nn.Tanh()).to(device)

    def update_symbol(self, s, z):
        # Similarity between current latent state z and symbol prototype
        # Ensure z has batch dim for unsqueeze if needed, but here z is usually 1D in training loop
        z_comp = z if z.dim() > 0 else z.unsqueeze(0)
        
        sim = torch.cosine_similarity(z_comp.unsqueeze(0), s.prototype.unsqueeze(0) + 1e-6).item()
        sim = max(0, sim) # ReLU activation logic
        
        # Leaky integrator for activation
        s.activation = s.persistence * s.activation + (1 - s.persistence) * sim
        
        # Hebbian-like update
        if s.activation > 0.5:
             diff = (z.detach() - s.prototype)
             s.prototype += self.lr * s.activation * diff

    def forward(self, vec, comm_in, raw=None):
        inp = torch.cat([vec, comm_in])
        z = self.encoder(inp)
        
        # --- Bottom-Up Resonance ---
        # 1. Low Level
        for s in self.low.values(): self.update_symbol(s, z)
        if raw and raw["NEAR_WALL"]: self.low["WALL"].activation = 1.0
        
        # 2. Mid Level (Association)
        self.mid["GOAL"].activation = self.low["FOOD"].activation
        self.mid["THREAT"].activation = self.low["WALL"].activation
        for s in self.mid.values(): self.update_symbol(s, z)
        
        # 3. High Level (Abstract)
        self.high["SURVIVAL"].activation = (self.mid["GOAL"].activation + (1.0 - self.mid["THREAT"].activation)) / 2
        for s in self.high.values(): self.update_symbol(s, z)

        comm = self.comm_out(z)
        
        return z, {**self.low, **self.mid, **self.high}, comm

# ============================================================
# 3. Policy & World Model
# ============================================================
class Policy(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4) 
        ).to(device)
    def forward(self, z): return self.fc(z)

class SymbolTransitionModel(nn.Module):
    def __init__(self, symbol_dim, action_dim=4):
        super().__init__()
        # Predicts z_{t+1} given z_t and a_t
        self.fc = nn.Sequential(
            nn.Linear(symbol_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, symbol_dim),
            nn.Tanh()
        ).to(device)

    def forward(self, symbol_vec, action):
        # ----------------------------------------------------
        # FIX: Robustly handle 1D (single) and 2D (batched) inputs
        # ----------------------------------------------------
        is_batched = symbol_vec.dim() > 1
        
        # If unbatched [D], make it [1, D]
        x = symbol_vec if is_batched else symbol_vec.unsqueeze(0)
        
        batch_size = x.size(0)
        action_onehot = torch.zeros((batch_size, 4), device=device)

        # Handle Action (Scalar, 1D tensor, or Batch)
        if action.dim() == 0:
            # Scalar action
            action_onehot[:, action] = 1.0
        elif action.dim() == 1 and action.size(0) == 1 and batch_size > 1:
            # Single action tensor broadcast
            action_onehot[:, action[0]] = 1.0
        else:
            # Batch of actions (or single if B=1)
            # If action is shape [B], we need unsqueeze to [B, 1] for scatter
            indices = action.unsqueeze(1) if action.dim() == 1 else action
            action_onehot.scatter_(1, indices, 1.0)
            
        inp = torch.cat([x, action_onehot], dim=1)
        out = self.fc(inp)
        
        # Return to original dimensionality
        if not is_batched:
            return out.squeeze(0)
        return out

# ============================================================
# 4. Planning & Ethics
# ============================================================
class MultiAgentMCTSPlanner:
    def __init__(self, transition_model, horizon=3, simulations=10):
        self.model = transition_model
        self.horizon = horizon
        self.simulations = simulations

    def rollout(self, symbol_vec):
        # Prepare batched input for the model [1, D]
        symbol_vec = symbol_vec.detach().unsqueeze(0) 
        
        best_action = random.randint(0, 3)
        best_val = -float('inf')

        for start_action in range(4):
            avg_score = 0
            for _ in range(self.simulations):
                curr_z = symbol_vec
                
                # First step
                curr_z = self.model(curr_z, torch.tensor([start_action], device=device))
                
                # Rollout
                score = 0
                for h in range(self.horizon - 1):
                    rand_act = torch.randint(0, 4, (1,), device=device)
                    curr_z = self.model(curr_z, rand_act)
                    # Heuristic: maximize activity/survival (proxy)
                    score += torch.mean(curr_z).item() 
                avg_score += score
            
            if avg_score > best_val:
                best_val = avg_score
                best_action = start_action
                
        return best_action

class EthicalAI:
    def filter(self, logits, symbols, others_symbols):
        probs = torch.softmax(logits, dim=0)
        mask = torch.ones_like(probs)

        # 1. Self-Preservation Constraint (System 2 Override)
        if symbols["THREAT"].activation > 0.8:
            # Dampen probabilities generically if threatened (rudimentary)
            pass 

        # 2. Altruism Constraint
        for other in others_symbols:
            if other["SURVIVAL"].activation > 0.9:
                # If neighbor is struggling, reduce aggressive moves
                mask[0] *= 0.5 
                mask[1] *= 0.5

        probs = probs * mask
        return torch.multinomial(probs, 1).item()

# ============================================================
# 5. Training Loop
# ============================================================
def train():
    env = MultiSnakeEnv(num_agents=NUM_AGENTS, grid_size=GRID_SIZE)
    
    # Initialize Agents
    agents = []
    for _ in range(NUM_AGENTS):
        agents.append({
            "resonance": HierarchicalResonance(6, PROTOTYPE_DIM, COMM_DIM),
            "policy": Policy(PROTOTYPE_DIM),
            "memory": SymbolMemory(),
            "comm": torch.zeros(COMM_DIM, device=device)
        })

    transition_model = SymbolTransitionModel(PROTOTYPE_DIM)
    planner = MultiAgentMCTSPlanner(transition_model, horizon=HORIZON, simulations=MCTS_SIMS)
    ethics = EthicalAI()

    params = list(transition_model.parameters())
    for a in agents:
        params += list(a["resonance"].parameters())
        params += list(a["policy"].parameters())
    
    optimizer = optim.Adam(params, lr=LR)
    mse_loss = nn.MSELoss()

    print("Starting Neuro-Symbolic Training...")
    
    for ep in range(1, 101):
        states = env.reset()
        for a in agents: a["memory"].working.clear()
        
        ep_rewards = np.zeros(NUM_AGENTS)
        steps = 0
        
        while not all(env.dead) and steps < 100:
            steps += 1
            actions = []
            agent_data = [] 
            
            # --- Decision Phase ---
            for i, a in enumerate(agents):
                if env.dead[i]:
                    actions.append(0)
                    agent_data.append(None)
                    continue

                others_comm = torch.zeros(COMM_DIM, device=device)
                count = 0
                for j in range(NUM_AGENTS):
                    if i != j and not env.dead[j]:
                        others_comm += agents[j]["comm"]
                        count += 1
                if count > 0: others_comm /= count

                vec = torch.tensor(states[i][0], device=device)
                
                # 1. Perception
                z, symbols, comm_out = a["resonance"](vec, others_comm, states[i][1])
                a["comm"] = comm_out.detach() 

                # 2. Policy (System 1)
                logits = a["policy"](z)
                
                # 3. Planning (MCTS) or Ethics (System 2)
                if random.random() < 0.2:
                    action = planner.rollout(z)
                else:
                    other_syms = [agent_data[j]["symbols"] for j in range(len(agent_data)) if agent_data[j] is not None]
                    action = ethics.filter(logits, symbols, other_syms)

                actions.append(action)
                agent_data.append({
                    "z": z,
                    "logits": logits,
                    "action": action,
                    "symbols": symbols
                })

            # --- Environment Step ---
            next_states, rewards, dones = env.step(actions)

            # --- Learning Phase ---
            optimizer.zero_grad()
            total_loss = 0
            
            for i, data in enumerate(agent_data):
                if data is None: continue
                
                # Target Z (Future Concept)
                next_vec = torch.tensor(next_states[i][0], device=device)
                next_z, _, _ = agents[i]["resonance"](next_vec, agents[i]["comm"], next_states[i][1])
                next_z = next_z.detach() 

                # A. Transition Model Loss
                # Data["z"] is 1D, but corrected model handles it
                pred_next_z = transition_model(data["z"], torch.tensor(data["action"], device=device))
                loss_trans = mse_loss(pred_next_z, next_z)

                # B. Policy Loss (Curiosity driven)
                r_intrinsic = 0.1 * agents[i]["memory"].novelty()
                r_total = rewards[i] + r_intrinsic

                log_prob = torch.log_softmax(data["logits"], dim=0)[data["action"]]
                loss_policy = -log_prob * r_total

                agents[i]["memory"].update(data["symbols"])
                total_loss += loss_policy + loss_trans
                ep_rewards[i] += r_total

            if total_loss != 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            states = next_states
            
            if ep % 20 == 0 or ep == 1:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Episode {ep} Step {steps}")
                env.render()
                if not env.dead[0] and agent_data[0]:
                    s = agent_data[0]["symbols"]
                    print(f"Ag0: FOOD={s['FOOD'].activation:.2f} THREAT={s['THREAT'].activation:.2f} SURVIVAL={s['SURVIVAL'].activation:.2f}")
                time.sleep(0.05)

        print(f"Ep {ep:03d} | Avg Reward: {np.mean(ep_rewards):.3f} | Steps: {steps}")

if __name__ == "__main__":
    train()
