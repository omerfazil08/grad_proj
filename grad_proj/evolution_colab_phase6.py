# evolution_colab_phase6.py
# Phase 6: Bitwise Engine + Sequential Targets + ISLAND MODEL + Diversity Fix
# Solves "Clonal Collapse" by maintaining spatial isolation and forcing diversity on resets.

import random
import time
import statistics
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import copy

# Try importing simplifier
try:
    import simplifier_phase14
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False
    print("⚠️ Warning: simplifier_phase14 not found. Defragmentation disabled.")

# --- Configuration ---
class BitwiseConfig:
    def __init__(self, num_gates, pop_size, generations, elitism, tournament_k, 
                 base_mut, min_mut, p_choose_primitive, log_every, 
                 record_history, seed, size_penalty_lambda, parallel=True, num_islands=5):
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
        self.num_islands = num_islands

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

# --- Init ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def init_population(num_inputs, cfg, size):
    pop = []
    for _ in range(size):
        indiv = [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]
        pop.append(indiv)
    return pop

# --- Mutation ---
def mutate(ind, num_inputs, rate, cfg, locked_count):
    new_ind = []
    for i, gate in enumerate(ind):
        if i < locked_count:
            new_ind.append(gate) # Locked gates are immutable
        elif random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# --- Eval ---
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

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target_idx):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    num_gates = len(individual)
    num_targets = len(packed_targets)
    max_score_per_col = mask.bit_count()
    
    # 1. Check Past Solved Outputs (Penalty if broken)
    penalty = 0
    for i in range(current_target_idx):
        if i in solved_output_map:
            gate_idx = solved_output_map[i]
            sig_idx = num_inputs + gate_idx
            out_val = signals[sig_idx] if sig_idx < len(signals) else 0
            diff = out_val ^ packed_targets[i]
            if diff != 0: penalty += 50000
    
    # 2. Score Current Target
    if current_target_idx in solved_output_map:
        gate_idx = solved_output_map[current_target_idx]
        sig_idx = num_inputs + gate_idx
    else:
        # Standard convention: Output i is (Total_Gates - Num_Outputs + i)
        sig_idx = num_inputs + (num_gates - num_targets + current_target_idx)
    
    out_val = signals[sig_idx] if sig_idx < len(signals) else 0
    diff = out_val ^ packed_targets[current_target_idx]
    current_score = max_score_per_col - diff.bit_count()
    
    return current_score - penalty, current_score

# --- Parallel ---
_PE_data = {}
def _init_pool(inputs, targets, mask, n_in, solved_map, curr_target):
    _PE_data['inputs'] = inputs
    _PE_data['targets'] = targets
    _PE_data['mask'] = mask
    _PE_data['n_in'] = n_in
    _PE_data['solved_map'] = solved_map
    _PE_data['curr_target'] = curr_target

def _eval_wrapper(ind):
    return fitness_bitwise(ind, _PE_data['inputs'], _PE_data['targets'], _PE_data['mask'], 
                           _PE_data['n_in'], _PE_data['solved_map'], _PE_data['curr_target'])

# --- Defragmentation ---
def convert_to_string_format(individual, num_inputs):
    string_gates = []
    for i, gate in enumerate(individual):
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs: str_inputs.append(f"A{inp_idx}")
            else: str_inputs.append(f"G{inp_idx - num_inputs}")
        string_gates.append({'name': f"G{i}", 'type': gate['type'], 'inputs': str_inputs, 'output': f"G{i}"})
    return string_gates

def defragment_genome(best_ind, solved_indices, solved_output_map, num_inputs, num_gates, num_outputs):
    if not HAS_SIMPLIFIER: return best_ind, 0, {}

    keep_indices = set()
    
    for out_i in range(num_outputs):
        if out_i in solved_indices:
            if out_i in solved_output_map:
                gate_idx = solved_output_map[out_i]
            else:
                gate_idx = len(best_ind) - num_outputs + out_i
            
            stack = [gate_idx]
            keep_indices.add(gate_idx)
            while stack:
                curr = stack.pop()
                g = best_ind[curr]
                for inp in g['inputs']:
                    if inp >= num_inputs:
                        src_idx = inp - num_inputs
                        if src_idx not in keep_indices:
                            keep_indices.add(src_idx)
                            stack.append(src_idx)
                            
    sorted_keep = sorted(list(keep_indices))
    remap = {old: new for new, old in enumerate(sorted_keep)}
    
    new_genome = []
    for old_idx in sorted_keep:
        gate = best_ind[old_idx].copy()
        new_inputs = []
        for inp in gate['inputs']:
            if inp >= num_inputs:
                src = inp - num_inputs
                new_inputs.append(num_inputs + remap.get(src, 0))
            else:
                new_inputs.append(inp)
        gate['inputs'] = new_inputs
        gate['name'] = f"G{len(new_genome)}"
        new_genome.append(gate)
        
    locked_count = len(new_genome)
    
    # Note: We do NOT fill the rest here. The main loop fills the tails randomly
    # to ensure diversity. We just return the locked prefix.
    
    final_output_map = {}
    for out_i in solved_indices:
        if out_i in solved_output_map:
            old = solved_output_map[out_i]
        else:
            old = len(best_ind) - num_outputs + out_i
        final_output_map[out_i] = remap.get(old, 0)
        
    return new_genome, locked_count, final_output_map

# --- Main Island Loop ---
def evolve_islands(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    # 1. Initialize Islands
    num_islands = cfg.num_islands
    island_pop_size = cfg.pop_size // num_islands
    
    print(f"🏝️ Phase 6: {num_islands} Islands x {island_pop_size} Pop. Target: {num_outputs} outputs.")
    
    islands = []
    for _ in range(num_islands):
        islands.append(init_population(num_inputs, cfg, island_pop_size))
        
    solved_mask = [False] * num_outputs
    solved_output_map = {}
    locked_count = 0
    current_target = 0
    
    history = {'gen': [], 'best': []}
    
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target))

    try:
        for gen in range(cfg.generations):
            # Batch all individuals for parallel evaluation
            flat_pop = [ind for island in islands for ind in island]
            
            if pool:
                results = pool.map(_eval_wrapper, flat_pop)
            else:
                results = [_eval_wrapper(ind) for ind in flat_pop]
            
            # Process results per island
            offset = 0
            global_best_val = -1e9
            global_best_ind = None
            global_best_raw_score = 0
            
            new_islands = []
            
            for i in range(num_islands):
                # Extract island data
                island_results = results[offset : offset + island_pop_size]
                island_pop = islands[i]
                offset += island_pop_size
                
                # Unpack scores
                scalars = [r[0] for r in island_results] # Fitness with penalty
                raw_scores = [r[1] for r in island_results] # Raw score for current target
                
                # Check Local Best
                local_best_val = max(scalars)
                local_best_idx = scalars.index(local_best_val)
                
                # Update Global Best (Tracking)
                if local_best_val > global_best_val:
                    global_best_val = local_best_val
                    global_best_ind = island_pop[local_best_idx]
                    global_best_raw_score = raw_scores[local_best_idx]
                
                # --- Evolution Step (Standard Elitism + Tournament) ---
                new_pop = []
                sorted_pop = [x for _, x in sorted(zip(scalars, island_pop), key=lambda p: p[0], reverse=True)]
                new_pop.extend([g[:] for g in sorted_pop[:cfg.elitism]])
                
                while len(new_pop) < island_pop_size:
                    # Simple random parent selection for speed
                    p1 = random.choice(island_pop)
                    p2 = random.choice(island_pop)
                    c1, c2 = crossover(p1, p2)
                    new_pop.append(mutate(c1, num_inputs, cfg.base_mut, cfg, locked_count))
                    if len(new_pop) < island_pop_size:
                        new_pop.append(mutate(c2, num_inputs, cfg.base_mut, cfg, locked_count))
                
                new_islands.append(new_pop)
            
            islands = new_islands

            # --- MIGRATION (Ring Topology) ---
            if gen % 50 == 0:
                for i in range(num_islands):
                    target_i = (i + 1) % num_islands
                    # Best of Island i overwrites Worst of Island i+1
                    # (Note: islands are already roughly sorted by elites at 0 due to elitism copy, 
                    # but new children are appended. Let's just grab index 0 as approximate best)
                    islands[target_i][-1] = copy.deepcopy(islands[i][0])

            # --- Check Global Success ---
            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Target #{current_target+1}: {global_best_raw_score}/{max_score_per_col} | Locked={locked_count}")

            if global_best_raw_score == max_score_per_col:
                print(f"🎉 Output #{current_target+1} SOLVED at Gen {gen}!")
                solved_mask[current_target] = True
                
                # 1. Defragment using the BEST genome found across all islands
                print(f"   >> Defragmenting genome...")
                solved_indices = [idx for idx, x in enumerate(solved_mask) if x]
                
                # Returns only the locked prefix
                locked_prefix, new_lock_count, new_map = defragment_genome(
                    global_best_ind, solved_indices, solved_output_map, num_inputs, cfg.num_gates, num_outputs
                )
                
                print(f"   >> New Locked Gates: {new_lock_count}")
                locked_count = new_lock_count
                solved_output_map = new_map
                
                current_target += 1
                if current_target >= num_outputs:
                    print("✅ ALL OUTPUTS SOLVED!")
                    return global_best_ind, None, {}, history

                # 2. Update Pool for next target
                if pool:
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target))
                
                # 3. GLOBAL RESET with DIVERSITY (The Fix)
                # Re-initialize all islands. 
                # Everyone gets the Locked Prefix, but a BRAND NEW Random Tail.
                print("   >> 🌊 Global Population Reset (Restoring Diversity)...")
                islands = []
                for _ in range(num_islands):
                    island_pop = []
                    for _ in range(island_pop_size):
                        indiv = []
                        # Copy locked prefix
                        for g in locked_prefix:
                            indiv.append(g.copy())
                        # Generate random tail
                        for k in range(locked_count, cfg.num_gates):
                            limit = num_inputs + k
                            gtype = random.choice(PRIMITIVES + MACROS)
                            _, arity = GATE_OPS[gtype]
                            ins = [random.randint(0, limit - 1) for _ in range(arity)]
                            indiv.append({'name': f"G{k}", 'type': gtype, 'inputs': ins})
                        island_pop.append(indiv)
                    islands.append(island_pop)
                continue

    finally:
        if pool: pool.close(); pool.join()

    # Return best found so far
    return global_best_ind, None, {}, history