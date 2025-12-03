# evolution_colab_phase5.py
# Phase 5 Revised: Holistic Hybrid Engine
# Integrates Bitwise Macros (Phase 5) with Holistic Fitness & Locking (Phase 3.7)

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
                 num_gates=50, 
                 pop_size=2000, 
                 generations=5000, # Total generations (not per bit)
                 elitism=50, 
                 tournament_k=10, 
                 base_mut=0.05, 
                 min_mut=0.005, 
                 p_choose_primitive=0.50, 
                 log_every=50, 
                 seed=42, 
                 parallel=True):
        self.num_gates = num_gates
        self.pop_size = pop_size
        self.generations = generations
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
# ... (Same as before) ...
def b_AND(a, b, mask): return a & b
def b_OR(a, b, mask):  return a | b
def b_XOR(a, b, mask): return a ^ b
def b_NOT(a, mask):    return (~a) & mask
def b_NAND(a, b, mask): return (~(a & b)) & mask
def b_NOR(a, b, mask):  return (~(a | b)) & mask
def b_XNOR(a, b, mask): return (~(a ^ b)) & mask
def b_MUX2(s, a, b, mask): return (((~s) & mask) & a) | (s & b)
def b_HALF_SUM(a, b, mask): return a ^ b
def b_HALF_CARRY(a, b, mask): return a & b
def b_FULL_SUM(a, b, c, mask): return a ^ b ^ c
def b_FULL_CARRY(a, b, c, mask): return (a & b) | (a & c) | (b & c)
def b_EQ1(a, b, mask): return (~(a ^ b)) & mask
def b_GT1(a, b, mask): return (a & ((~b) & mask))

GATE_OPS = {
    "AND": (b_AND, 2), "OR": (b_OR, 2), "XOR": (b_XOR, 2), "XOR2": (b_XOR, 2),
    "NOT": (b_NOT, 1), "NAND": (b_NAND, 2), "NOR": (b_NOR, 2), "XNOR": (b_XNOR, 2),
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
    if random.random() < p_prim:
        gtype = random.choice(PRIMITIVES)
    else:
        gtype = random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'type': gtype, 'inputs': ins}

def init_population(num_inputs, cfg):
    pop = []
    for _ in range(cfg.pop_size):
        ind = [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]
        pop.append(ind)
    return pop

def pack_truth_table(inputs, num_inputs):
    num_rows = len(inputs)
    mask = (1 << num_rows) - 1
    packed = [0] * num_inputs
    for r, row in enumerate(inputs):
        for c, bit in enumerate(row):
            if bit: packed[c] |= (1 << r)
    return packed, mask

def pack_targets(targets):
    return [sum((bit << r) for r, bit in enumerate(col)) for col in targets]

# --- Trace Active Logic (For Locking) ---
def trace_active_gates(individual, output_indices, num_inputs):
    active = set()
    # Only trace outputs that are being locked NOW
    stack = list(output_indices)
    while stack:
        curr = stack.pop()
        if curr < num_inputs: continue
        g_idx = curr - num_inputs
        if g_idx in active: continue
        active.add(g_idx)
        for inp in individual[g_idx]['inputs']:
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

def fitness_holistic(individual, packed_inputs, packed_targets, mask, num_inputs):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    # Outputs are the LAST N gates
    num_outputs = len(packed_targets)
    output_signals = signals[-num_outputs:]
    
    scores = []
    for i, target in enumerate(packed_targets):
        diff = output_signals[i] ^ target
        scores.append(mask.bit_count() - diff.bit_count())
    return scores

# --- Parallel Worker ---
_PE_inputs = None
_PE_targets = None
_PE_mask = None
_PE_n_in = None

def _init_pool(inputs, targets, mask, n_in):
    global _PE_inputs, _PE_targets, _PE_mask, _PE_n_in
    _PE_inputs = inputs
    _PE_targets = targets
    _PE_mask = mask
    _PE_n_in = n_in

def _eval_wrapper(ind):
    scores = fitness_holistic(ind, _PE_inputs, _PE_targets, _PE_mask, _PE_n_in)
    return sum(scores), scores

# --- Ops ---
def mutate(ind, num_inputs, rate, cfg, locked_indices):
    new_ind = []
    for i, gate in enumerate(ind):
        if i in locked_indices:
            new_ind.append(gate)
        elif random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# ==============================================================================
# 4. MAIN EVOLUTION LOOP
# ==============================================================================
def evolve_incremental_phase5(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    print(f"Phase 5 Revised: Holistic Fitness + Macros + Incremental Locking")
    population = init_population(num_inputs, cfg)
    
    locked_indices = set()
    solved_mask = [False] * num_outputs
    
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_targets, mask, num_inputs))

    try:
        for gen in range(cfg.generations):
            if pool:
                results = pool.map(_eval_wrapper, population)
                scalars = [r[0] for r in results]
                breakdowns = [r[1] for r in results]
            else:
                scalars, breakdowns = [], []
                for ind in population:
                    sc = fitness_holistic(ind, packed_inputs, packed_targets, mask, num_inputs)
                    scalars.append(sum(sc))
                    breakdowns.append(sc)
            
            # Stats
            best_val = max(scalars)
            best_idx = scalars.index(best_val)
            best_bkd = breakdowns[best_idx]
            best_ind = population[best_idx]
            
            # Check Solved
            new_solve = False
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    # Lock logic for this output
                    # Output gate is num_inputs + (total_gates - num_outputs + i)
                    abs_gate_idx = num_inputs + (cfg.num_gates - num_outputs + i)
                    active = trace_active_gates(best_ind, abs_gate_idx, num_inputs)
                    print(f"   🔒 Locking {len(active)} gates for Output {i+1}")
                    locked_indices.update(active)
                    new_solve = True
            
            # If new lock, force lock onto population (Ratchet)
            if new_solve:
                for i in range(len(population)):
                    for l_idx in locked_indices:
                        population[i][l_idx] = copy.deepcopy(best_ind[l_idx])

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | {best_bkd} | Locked: {len(locked_indices)}")
                
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[best_idx]

            # Reproduction
            new_pop = []
            sorted_indices = sorted(range(len(scalars)), key=lambda k: scalars[k], reverse=True)
            for i in range(cfg.elitism):
                new_pop.append(population[sorted_indices[i]])
                
            while len(new_pop) < cfg.pop_size:
                p1 = population[random.choice(range(len(population)))]
                p2 = population[random.choice(range(len(population)))]
                c1, c2 = crossover(p1, p2)
                
                rate = cfg.base_mut * (1 - gen/cfg.generations) + cfg.min_mut
                new_pop.append(mutate(c1, num_inputs, rate, cfg, locked_indices))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, rate, cfg, locked_indices))
            
            # Enforce Locks on Children
            for i in range(len(new_pop)):
                 for l_idx in locked_indices:
                    new_pop[i][l_idx] = best_ind[l_idx] # Copy from best known logic
            
            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    return population[best_idx]

def convert_to_string_format(individual, num_inputs):
    return [] # Placeholder as VHDL export is disabled for this test