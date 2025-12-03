# evolution_colab_phase4.py
# Phase 4: Divide & Conquer Engine (Single-Objective Bitwise)
# Focused on solving ONE output bit as fast as possible.

import random
import time
import statistics
from multiprocessing import Pool, cpu_count

# --- Configuration ---
class Phase4Config:
    def __init__(self, num_gates, pop_size, generations, elitism, tournament_k, 
                 base_mut, min_mut, p_choose_primitive, log_every, 
                 record_history, seed, parallel=True):
        self.num_gates = num_gates
        self.pop_size = pop_size
        self.generations = generations
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.base_mut = base_mut
        self.min_mut = min_mut
        self.p_choose_primitive = p_choose_primitive
        self.log_every = log_every
        self.record_history = record_history
        self.seed = seed
        self.parallel = parallel

# --- Bitwise Logic Gates ---
def b_AND(a, b, mask): return a & b
def b_OR(a, b, mask):  return a | b
def b_XOR(a, b, mask): return a ^ b
def b_NOT(a, mask):    return (~a) & mask
def b_NAND(a, b, mask): return (~(a & b)) & mask
def b_NOR(a, b, mask):  return (~(a | b)) & mask
def b_XNOR(a, b, mask): return (~(a ^ b)) & mask
def b_MUX2(s, a, b, mask): return (((~s) & mask) & a) | (s & b)

GATE_OPS = {
    "AND": (b_AND, 2), "OR": (b_OR, 2), "XOR": (b_XOR, 2), "XOR2": (b_XOR, 2),
    "NOT": (b_NOT, 1), 
    "NAND": (b_NAND, 2), "NOR": (b_NOR, 2), "XNOR": (b_XNOR, 2), "XNOR2": (b_XNOR, 2),
    "MUX2": (b_MUX2, 3)
}
PRIMITIVES = ["AND", "OR", "XOR", "NOT", "NAND", "NOR"]
MACROS = ["MUX2"]

# --- Helpers ---
def pack_truth_table(inputs, num_inputs):
    num_rows = len(inputs)
    mask = (1 << num_rows) - 1
    packed = [0] * num_inputs
    for row_idx, row in enumerate(inputs):
        for col_idx, bit in enumerate(row):
            if bit: packed[col_idx] |= (1 << row_idx)
    return packed, mask

def pack_target_column(target_col):
    """Packs a SINGLE output column into one integer."""
    val = 0
    for r, bit in enumerate(target_col):
        if bit: val |= (1 << r)
    return val

# --- Genome ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'type': gtype, 'inputs': ins}

def init_population(num_inputs, cfg):
    pop = []
    for _ in range(cfg.pop_size):
        ind = [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]
        pop.append(ind)
    return pop

# --- Evaluation ---
def evaluate_bitwise(individual, packed_inputs, mask):
    signals = list(packed_inputs)
    for gate in individual:
        gtype = gate['type']
        ins = gate['inputs']
        func, arity = GATE_OPS[gtype]
        
        # Fast lookup
        vals = [signals[i] for i in ins] # Indices guaranteed valid by construction
        
        if arity == 2: res = func(vals[0], vals[1], mask)
        elif arity == 1: res = func(vals[0], mask)
        elif arity == 3: res = func(vals[0], vals[1], vals[2], mask)
        signals.append(res)
    return signals

def fitness_single_target(individual, packed_inputs, target_val, mask):
    # Eval full circuit
    signals = evaluate_bitwise(individual, packed_inputs, mask)
    # Output is the LAST gate
    out_val = signals[-1]
    # Hamming distance
    diff = out_val ^ target_val
    return mask.bit_count() - diff.bit_count()

# --- Parallel ---
_PE_data = {}
def _init_pool(inputs, target, mask):
    _PE_data['inputs'] = inputs
    _PE_data['target'] = target
    _PE_data['mask'] = mask

def _eval_wrapper(ind):
    return fitness_single_target(ind, _PE_data['inputs'], _PE_data['target'], _PE_data['mask'])

# --- Evolution Ops ---
def mutate(ind, num_inputs, rate, cfg):
    new_ind = []
    for i, gate in enumerate(ind):
        if random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# --- Main Loop (Single Objective) ---
def evolve_single_output(num_inputs, inputs_list, target_col, cfg, label="OutX"):
    random.seed(cfg.seed)
    
    # 1. Pack Data
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_target = pack_target_column(target_col)
    max_score = mask.bit_count()
    
    print(f"[{label}] Evolving 1-bit target (Max={max_score})...")
    
    population = init_population(num_inputs, cfg)
    
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_target, mask))

    try:
        for gen in range(cfg.generations):
            # Eval
            if pool:
                scores = pool.map(_eval_wrapper, population)
            else:
                scores = [fitness_single_target(ind, packed_inputs, packed_target, mask) for ind in population]
            
            # Stats
            best_val = max(scores)
            best_idx = scores.index(best_val)
            
            if gen % cfg.log_every == 0:
                print(f"   Gen {gen:4d}: {best_val}/{max_score}")
                
            if best_val == max_score:
                print(f"   ✅ {label} SOLVED at Gen {gen}!")
                return population[best_idx]

            # Elitism
            new_pop = []
            sorted_pop = [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]
            new_pop.extend(sorted_pop[:cfg.elitism])
            
            # Reproduction
            while len(new_pop) < cfg.pop_size:
                # Tournament
                p1 = population[random.choice(range(len(population)))] # Simple random selection is often fast enough
                p2 = population[random.choice(range(len(population)))] # or impl tournament if stuck
                
                c1, c2 = crossover(p1, p2)
                
                rate = cfg.base_mut * (1 - gen/cfg.generations) + cfg.min_mut
                new_pop.append(mutate(c1, num_inputs, rate, cfg))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, rate, cfg))
            
            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    print(f"   ⚠️ {label} not perfectly solved. Best: {best_val}/{max_score}")
    return sorted_pop[0]

# --- Conversion ---
def convert_single_to_string(individual, num_inputs):
    """
    Converts internal ints to strings ("A0", "G5") assuming this circuit is standalone.
    The MAIN script will handle re-indexing when merging.
    """
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