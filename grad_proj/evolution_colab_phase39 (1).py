# evolution_colab_phase39.py
# Phase 3.9: Bitwise Engine + DEFRAGMENTATION (Relocation)
# Fixes "Spaghetti Locking" by moving solved logic to the start of the genome (G0, G1...)
# and freeing up the rest of the board.

import random
import time
import statistics
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# --- Configuration ---
class BitwiseConfig:
    def __init__(self, num_gates, pop_size, generations, elitism, tournament_k, 
                 base_mut, min_mut, p_choose_primitive, log_every, 
                 record_history, seed, size_penalty_lambda, parallel=True):
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
        self.size_penalty_lambda = size_penalty_lambda
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
def gate_name(idx): return f"G{idx}"

def pack_truth_table(inputs, num_inputs):
    num_rows = len(inputs)
    mask = (1 << num_rows) - 1
    packed = [0] * num_inputs
    for row_idx, row in enumerate(inputs):
        for col_idx, bit in enumerate(row):
            if bit: packed[col_idx] |= (1 << row_idx)
    return packed, mask

def pack_targets(targets):
    packed = []
    for col in targets:
        val = 0
        for r, bit in enumerate(col):
            if bit: val |= (1 << r)
        packed.append(val)
    return packed

# --- Genome & Init ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    # Inputs: 0..(num_inputs-1) are pins, num_inputs..limit-1 are gates
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def init_population(num_inputs, cfg):
    pop = []
    for _ in range(cfg.pop_size):
        indiv = [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]
        pop.append(indiv)
    return pop

# --- Mutation (Respects Locked Count) ---
def mutate(ind, num_inputs, rate, cfg, locked_count):
    new_ind = []
    for i, gate in enumerate(ind):
        # If this gate index is within the "Locked Zone" (0 to locked_count-1), preserve it.
        if i < locked_count:
            new_ind.append(gate)
        elif random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# --- Evaluation (Supports Mapped Outputs) ---
def evaluate_bitwise(individual, packed_inputs, mask, num_inputs):
    signals = list(packed_inputs)
    for gate in individual:
        gtype = gate['type']
        ins = gate['inputs']
        func, arity = GATE_OPS[gtype]
        vals = [signals[i] if i < len(signals) else 0 for i in ins]
        if arity == 1: res = func(vals[0], mask)
        elif arity == 2: res = func(vals[0], vals[1], mask)
        elif arity == 3: res = func(vals[0], vals[1], vals[2], mask)
        signals.append(res)
    return signals

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs, solved_output_map):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    
    # Determine where to read outputs from
    # solved_output_map: {target_idx: gate_idx} -> Output i is at Gate X
    # Unsolved outputs: Read from standard position (End - num_unsolved + offset)? 
    # Simpler: Unsolved outputs are ALWAYS the last N gates.
    
    num_gates = len(individual)
    num_targets = len(packed_targets)
    scores = []
    
    for i, target in enumerate(packed_targets):
        if i in solved_output_map:
            # Locked output location
            gate_idx = solved_output_map[i]
            sig_idx = num_inputs + gate_idx
        else:
            # Floating output location (Last N gates)
            # e.g. 6 outputs. 0,1 locked. 2,3,4,5 are at -4, -3, -2, -1.
            # We need to map 'i' to 'unsolved_index'.
            # This is tricky. Let's simplify:
            # ALWAYS reserve the last N gates as "Candidate Outputs".
            # If Output 2 is unsolved, it reads from (End - N + 2).
            sig_idx = num_inputs + (num_gates - num_targets + i)
            
        if sig_idx < len(signals):
            out_val = signals[sig_idx]
        else:
            out_val = 0
            
        diff = out_val ^ target
        scores.append(mask.bit_count() - diff.bit_count())
    return scores

# --- Parallel Wrapper ---
_PE_data = {}
def _init_pool(inputs, targets, mask, n_in, solved_map):
    _PE_data['inputs'] = inputs
    _PE_data['targets'] = targets
    _PE_data['mask'] = mask
    _PE_data['n_in'] = n_in
    _PE_data['solved_map'] = solved_map

def _eval_wrapper(ind):
    scores = fitness_bitwise(ind, _PE_data['inputs'], _PE_data['targets'], _PE_data['mask'], _PE_data['n_in'], _PE_data['solved_map'])
    return sum(scores), scores

# --- CORE DEFRAGMENTATION LOGIC ---
def defragment_genome(individual, solved_indices, solved_output_map, num_inputs, num_gates, num_outputs):
    """
    1. Identifies active gates for all solved outputs.
    2. Remaps them to G0, G1, G2...
    3. Fills rest with random.
    4. Returns (new_genome, new_locked_count, new_solved_map)
    """
    
    # 1. Trace Dependencies
    keep_indices = set()
    
    # We need to check ALL solved outputs to ensure we keep shared logic
    for out_i in solved_indices:
        if out_i in solved_output_map:
            # Already locked location
            gate_idx = solved_output_map[out_i]
        else:
            # Currently at standard position (end of genome)
            gate_idx = len(individual) - num_outputs + out_i
            
        # BFS Trace
        stack = [gate_idx]
        keep_indices.add(gate_idx)
        while stack:
            curr = stack.pop()
            if curr >= len(individual): continue
            gate = individual[curr]
            for inp in gate['inputs']:
                if inp >= num_inputs: # It's a gate
                    src_idx = inp - num_inputs
                    if src_idx not in keep_indices:
                        keep_indices.add(src_idx)
                        stack.append(src_idx)

    # 2. Create Mapping (Old -> New)
    # Must sort to preserve topological order!
    sorted_old_indices = sorted(list(keep_indices))
    remap = {old: new for new, old in enumerate(sorted_old_indices)}
    
    new_genome = []
    
    # 3. Build Compressed Genome
    for old_idx in sorted_old_indices:
        gate = individual[old_idx].copy()
        # Remap inputs
        new_inputs = []
        for inp in gate['inputs']:
            if inp >= num_inputs:
                # It refers to a gate
                old_src = inp - num_inputs
                if old_src in remap:
                    new_inputs.append(num_inputs + remap[old_src])
                else:
                    # This shouldn't happen if tracing is correct, but fallback to 0
                    new_inputs.append(0) 
            else:
                # It refers to a primary input (A0..An), keep as is
                new_inputs.append(inp)
        
        gate['inputs'] = new_inputs
        # Update name for clarity
        gate['name'] = f"G{len(new_genome)}"
        new_genome.append(gate)
        
    locked_count = len(new_genome)
    
    # 4. Fill remainder with random gates
    # These are the "Fresh Breadboard" for the next output
    while len(new_genome) < num_gates:
        idx = len(new_genome)
        # Standard random gate generation
        # Note: We pass p_prim=0.5 as default; ideally pass from config
        # Re-implementing simplified random_gate logic here:
        gtype = random.choice(PRIMITIVES + MACROS)
        _, arity = GATE_OPS[gtype]
        limit = num_inputs + idx
        ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
        new_genome.append({'name': f"G{idx}", 'type': gtype, 'inputs': ins})
        
    # 5. Update Output Map
    new_solved_map = {}
    for out_i in solved_indices:
        # Where was it?
        if out_i in solved_output_map:
            old_loc = solved_output_map[out_i]
        else:
            old_loc = len(individual) - num_outputs + out_i
            
        new_loc = remap[old_loc]
        new_solved_map[out_i] = new_loc
        
    return new_genome, locked_count, new_solved_map


# --- Main Evolution Loop ---
def evolve_bitwise(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    print(f"Bitwise v3.9: Defragmentation & Relocation Active.")
    
    population = init_population(num_inputs, cfg)
    solved_mask = [False] * num_outputs
    solved_output_map = {} # {output_idx: gate_idx}
    locked_count = 0
    
    history = {'gen': [], 'best': [], 'mu': [], 'sigma': []}
    
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map))

    try:
        for gen in range(cfg.generations):
            # Eval
            if pool:
                results = pool.map(_eval_wrapper, population)
                scalars = [r[0] for r in results]
                breakdowns = [r[1] for r in results]
            else:
                scalars, breakdowns = [], []
                for ind in population:
                    sc = fitness_bitwise(ind, packed_inputs, packed_targets, mask, num_inputs, solved_output_map)
                    scalars.append(sum(sc))
                    breakdowns.append(sc)
            
            best_val = max(scalars)
            best_idx = scalars.index(best_val)
            best_bkd = breakdowns[best_idx]
            
            history['gen'].append(gen)
            history['best'].append(best_val)
            history['mu'].append(statistics.mean(scalars))
            history['sigma'].append(statistics.stdev(scalars) if len(scalars)>1 else 0)

            # Check Solved
            new_solve = False
            solved_indices = []
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    new_solve = True
                if solved_mask[i]:
                    solved_indices.append(i)
            
            # --- DEFRAGMENTATION TRIGGER ---
            if new_solve:
                print(f"   >> Defragmenting Genome (Moving solved logic to G0-Gk)...")
                best_ind = population[best_idx]
                
                new_genome, new_lock_cnt, new_map = defragment_genome(
                    best_ind, solved_indices, solved_output_map, num_inputs, cfg.num_gates, num_outputs
                )
                
                print(f"   >> New Locked Gate Count: {new_lock_cnt}/{cfg.num_gates}")
                
                # Update State
                locked_count = new_lock_cnt
                solved_output_map = new_map
                
                # Overwrite Population (Ratchet effect)
                for i in range(len(population)):
                    population[i] = [g.copy() for g in new_genome]
                
                # Update Pool Globals
                if pool:
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map))

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | Locked={locked_count} | {best_bkd}")
                
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[best_idx], best_bkd, {}, history

            # Reproduction
            new_pop = []
            # Elitism
            sorted_pop = [x for _, x in sorted(zip(scalars, population), key=lambda pair: pair[0], reverse=True)]
            new_pop.extend([g[:] for g in sorted_pop[:cfg.elitism]]) # Deep copy slices
            
            while len(new_pop) < cfg.pop_size:
                p1 = population[random.randint(0, len(population)-1)]
                p2 = population[random.randint(0, len(population)-1)]
                
                c1, c2 = crossover(p1, p2)
                
                # Mutation respects Locked Count (doesn't touch G0..Locked)
                new_pop.append(mutate(c1, num_inputs, cfg.base_mut, cfg, locked_count))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, cfg.base_mut, cfg, locked_count))
            
            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    return population[best_idx], best_bkd, {}, history

def convert_to_string_format(individual, num_inputs):
    string_gates = []
    for i, gate in enumerate(individual):
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs: str_inputs.append(f"A{inp_idx}")
            else: str_inputs.append(f"G{inp_idx - num_inputs}")
        string_gates.append({'name': f"G{i}", 'type': gate['type'], 'inputs': str_inputs, 'output': f"G{i}"})
    return string_gates