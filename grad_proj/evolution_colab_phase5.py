# evolution_colab_phase5.py
# Phase 5: The Hybrid Engine (Bitwise + Macros + Incremental Locking)
# Solves the scalability issue by evolving outputs sequentially (LSB -> MSB).

import random
import time
import statistics
import copy
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Phase5Config:
    def __init__(self, 
                 num_gates=60, 
                 pop_size=2000, 
                 generations_per_bit=2000, # How long to spend on EACH output
                 elitism=50, 
                 tournament_k=10, 
                 base_mut=0.05, 
                 min_mut=0.005, 
                 p_choose_primitive=0.60, # 60% AND/OR, 40% Macros
                 log_every=50, 
                 seed=42, 
                 parallel=True):
        self.num_gates = num_gates
        self.pop_size = pop_size
        self.generations_per_bit = generations_per_bit
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.base_mut = base_mut
        self.min_mut = min_mut
        self.p_choose_primitive = p_choose_primitive
        self.log_every = log_every
        self.seed = seed
        self.parallel = parallel

# ==============================================================================
# 2. BITWISE GATE LIBRARY (PRIMITIVES + MACROS)
# ==============================================================================
# All functions take (a, b, [c], mask) and return an integer.

# --- Primitives ---
def b_AND(a, b, mask): return a & b
def b_OR(a, b, mask):  return a | b
def b_XOR(a, b, mask): return a ^ b
def b_NOT(a, mask):    return (~a) & mask
def b_NAND(a, b, mask): return (~(a & b)) & mask
def b_NOR(a, b, mask):  return (~(a | b)) & mask
def b_XNOR(a, b, mask): return (~(a ^ b)) & mask

# --- Macros (The "Smart" Blocks) ---
# MUX2: If S=0 -> A, else -> B
def b_MUX2(s, a, b, mask): 
    return (((~s) & mask) & a) | (s & b)

# Half Adder (Sum, Carry) - We implement them as single-output gates for flexibility
def b_HALF_SUM(a, b, mask): return a ^ b
def b_HALF_CARRY(a, b, mask): return a & b

# Full Adder
def b_FULL_SUM(a, b, c, mask): return a ^ b ^ c
def b_FULL_CARRY(a, b, c, mask): return (a & b) | (a & c) | (b & c)

# Comparators
def b_EQ1(a, b, mask): return (~(a ^ b)) & mask # XNOR
def b_GT1(a, b, mask): return (a & ((~b) & mask)) # A > B is A=1, B=0

GATE_OPS = {
    "AND": (b_AND, 2), "OR": (b_OR, 2), "XOR": (b_XOR, 2), "XOR2": (b_XOR, 2),
    "NOT": (b_NOT, 1), "NAND": (b_NAND, 2), "NOR": (b_NOR, 2), "XNOR": (b_XNOR, 2),
    # Macros
    "MUX2": (b_MUX2, 3),
    "HALF_SUM": (b_HALF_SUM, 2), "HALF_CARRY": (b_HALF_CARRY, 2),
    "FULL_SUM": (b_FULL_SUM, 3), "FULL_CARRY": (b_FULL_CARRY, 3),
    "EQ1": (b_EQ1, 2), "GT1": (b_GT1, 2)
}

PRIMITIVES = ["AND", "OR", "XOR", "NOT", "NAND", "NOR"]
MACROS = ["MUX2", "HALF_SUM", "HALF_CARRY", "FULL_SUM", "FULL_CARRY", "EQ1", "GT1"]

# ==============================================================================
# 3. CORE EVOLUTION ENGINE
# ==============================================================================

def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    # Adaptive probability: Favor primitives or macros?
    if random.random() < p_prim:
        gtype = random.choice(PRIMITIVES)
    else:
        gtype = random.choice(MACROS)
        
    _, arity = GATE_OPS[gtype]
    
    # Inputs can be primary inputs (0..num_inputs-1) or previous gates
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'type': gtype, 'inputs': ins}

def init_population(num_inputs, cfg):
    # Standard random init
    pop = []
    for _ in range(cfg.pop_size):
        ind = [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]
        pop.append(ind)
    return pop

def pack_truth_table(inputs, num_inputs):
    """Packs truth table columns into integers for bitwise ops."""
    num_rows = len(inputs)
    mask = (1 << num_rows) - 1
    packed = [0] * num_inputs
    for r, row in enumerate(inputs):
        for c, bit in enumerate(row):
            if bit: packed[c] |= (1 << r)
    return packed, mask

def pack_target_column(target_col):
    val = 0
    for r, bit in enumerate(target_col):
        if bit: val |= (1 << r)
    return val

# --- Path Tracing (For Locking) ---
def trace_active_gates(individual, gate_idx, num_inputs):
    """
    Finds all gates that contribute to the output at `gate_idx`.
    """
    active = set()
    stack = [gate_idx]
    
    # Map gate_index -> actual gate dict is trivial (it's the list index)
    while stack:
        curr = stack.pop()
        # If curr is a primary input, ignore
        if curr < num_inputs:
            continue
            
        # Convert to list index (0-based)
        g_idx = curr - num_inputs
        if g_idx in active: continue
        
        # Mark active
        active.add(g_idx)
        gate = individual[g_idx]
        
        # Add inputs to stack
        for inp in gate['inputs']:
            stack.append(inp)
            
    return active

# --- Evaluation ---
def evaluate_bitwise(individual, packed_inputs, mask, num_inputs):
    signals = list(packed_inputs)
    for gate in individual:
        gtype = gate['type']
        ins = gate['inputs']
        func, arity = GATE_OPS[gtype]
        
        vals = [signals[i] for i in ins]
        
        if arity == 1: res = func(vals[0], mask)
        elif arity == 2: res = func(vals[0], vals[1], mask)
        elif arity == 3: res = func(vals[0], vals[1], vals[2], mask)
        
        signals.append(res)
    return signals

def fitness_single_target(individual, packed_inputs, target_val, mask, num_inputs, output_gate_idx):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    # Check the specific gate we designated as the output
    # Note: For incremental evolution, we might scan the LAST K gates to find the best one,
    # OR we might assign specific gates.
    # Phase 5 Strategy: The output for the current bit is assumed to be the LAST active gate.
    
    # Actually, let's allow the GA to use ANY gate as the output.
    # But for simplicity in the loop, let's assume the very last gate is the candidate for now.
    out_val = signals[output_gate_idx] # This is absolute index (num_inputs + gate_idx)
    
    diff = out_val ^ target_val
    return mask.bit_count() - diff.bit_count()

# --- Parallel Worker ---
_PE_inputs = None
_PE_target = None
_PE_mask = None
_PE_n_in = None
_PE_out_idx = None

def _init_pool(inputs, target, mask, n_in, out_idx):
    global _PE_inputs, _PE_target, _PE_mask, _PE_n_in, _PE_out_idx
    _PE_inputs = inputs
    _PE_target = target
    _PE_mask = mask
    _PE_n_in = n_in
    _PE_out_idx = out_idx

def _eval_wrapper(ind):
    # Calculate index of the last gate in this individual
    # Note: 'ind' is just a list of gates. 
    # The output signal index = num_inputs + len(ind) - 1
    last_gate_idx = _PE_n_in + len(ind) - 1
    return fitness_single_target(ind, _PE_inputs, _PE_target, _PE_mask, _PE_n_in, last_gate_idx)

# --- Evolution Operators ---
def mutate(ind, num_inputs, rate, cfg, locked_indices):
    new_ind = []
    for i, gate in enumerate(ind):
        # If locked, perfect copy
        if i in locked_indices:
            new_ind.append(gate)
            continue
            
        if random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

def crossover(p1, p2):
    if len(p1) < 2: return p1, p2
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# ==============================================================================
# 4. MAIN INCREMENTAL LOOP (The "Phase 5" Logic)
# ==============================================================================

# Replace the existing 'evolve_incremental_phase5' function with this:

def evolve_incremental_phase5(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    
    # 1. Pack Data
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = [pack_target_column(t) for t in targets_list]
    max_score = mask.bit_count()
    
    print(f"Phase 5 Hybrid Engine: Bitwise + Macros + Incremental Locking")
    print(f"Target: {num_inputs} Inputs -> {num_outputs} Outputs. Max Score/Bit: {max_score}")
    
    # Initialize Population
    population = init_population(num_inputs, cfg)
    
    # State tracking
    locked_indices = set()
    best_final_ind = None # Keeps the "Master Template" with correct locked gates
    
    # Iterate through outputs SEQUENTIALLY (0..N)
    for out_idx in range(num_outputs):
        current_target = packed_targets[out_idx]
        print(f"\n🎯 Evolving Output {out_idx+1}/{num_outputs}...")
        
        # Reset Pool for this target
        pool = None
        if cfg.parallel:
            pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                        initargs=(packed_inputs, current_target, mask, num_inputs, None))
            
        solved = False
        best_of_run = None
        
        # Generation Loop for CURRENT BIT
        for gen in range(cfg.generations_per_bit):
            if pool:
                scores = pool.map(_eval_wrapper, population)
            else:
                last_gate_idx = num_inputs + cfg.num_gates - 1
                scores = [fitness_single_target(ind, packed_inputs, current_target, mask, num_inputs, last_gate_idx) 
                          for ind in population]
            
            best_val = max(scores)
            best_ind = population[scores.index(best_val)]
            
            if gen % cfg.log_every == 0:
                print(f"   Gen {gen:4d} | Score: {best_val}/{max_score} | Locked: {len(locked_indices)}")
            
            if best_val == max_score:
                print(f"   ✅ Output {out_idx+1} SOLVED at Gen {gen}!")
                solved = True
                best_of_run = best_ind
                break
                
            # Standard GA steps
            new_pop = []
            # Elitism
            sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            for i in range(cfg.elitism):
                new_pop.append(population[sorted_indices[i]])
                
            while len(new_pop) < cfg.pop_size:
                p1 = population[random.choice(range(len(population)))]
                p2 = population[random.choice(range(len(population)))]
                c1, c2 = crossover(p1, p2)
                
                # Decay mutation rate
                rate = cfg.base_mut * (1 - gen/cfg.generations_per_bit) + cfg.min_mut
                new_pop.append(mutate(c1, num_inputs, rate, cfg, locked_indices))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, rate, cfg, locked_indices))
            
            # FORCE LOCKS (Critical Step)
            # Ensure no mutation accidentally modified a locked gate
            # We copy locks from the "Master Template" (best_final_ind) if it exists
            if best_final_ind:
                for i in range(len(new_pop)):
                    for lock_idx in locked_indices:
                        new_pop[i][lock_idx] = copy.deepcopy(best_final_ind[lock_idx])
            
            population = new_pop
            best_of_run = population[0] # Fallback
            
        if pool: pool.close(); pool.join()
        
        # --- LOCKING PHASE ---
        # 1. Trace which gates are used by the winner
        last_gate_abs_idx = num_inputs + len(best_of_run) - 1
        active_gates = trace_active_gates(best_of_run, last_gate_abs_idx, num_inputs)
        
        print(f"   🔒 Locking {len(active_gates)} active gates for Output {out_idx+1}")
        locked_indices.update(active_gates)
        
        # Store the winner as the new Master Template
        best_final_ind = copy.deepcopy(best_of_run)
        
        # 2. Ratchet Population (THE FIX)
        # Instead of cloning the winner 2000 times, we:
        # A. Keep the winner (Elitism)
        # B. Generate FRESH random individuals for the rest
        # C. Overwrite ONLY the locked gates in the fresh individuals with the Master Template
        
        print(f"   🔄 Ratcheting: Keeping locks, injecting FRESH diversity.")
        
        new_population = [copy.deepcopy(best_final_ind)] # Keep elite
        
        # Fill rest with fresh random genomes
        random_pop = init_population(num_inputs, cfg)
        
        for ind in random_pop:
            # Overwrite the locked positions with the correct logic
            for lock_idx in locked_indices:
                ind[lock_idx] = copy.deepcopy(best_final_ind[lock_idx])
            new_population.append(ind)
            
        # Trim to pop_size
        population = new_population[:cfg.pop_size]

    return best_final_ind

# --- String Converter for VHDL ---
def convert_to_string_format(individual, num_inputs):
    """Converts internal int-representation to 'A0', 'G5' strings for VHDL generator."""
    string_gates = []
    for i, gate in enumerate(individual):
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs:
                str_inputs.append(f"A{inp_idx}")
            else:
                gate_num = inp_idx - num_inputs
                str_inputs.append(f"G{gate_num}")
        
        string_gates.append({
            'name': f"G{i}",
            'type': gate['type'],
            'inputs': str_inputs,
            'output': f"G{i}"
        })
    return string_gates