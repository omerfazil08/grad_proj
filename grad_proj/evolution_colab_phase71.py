# evolution_colab_phase71.py
# Phase 7.1: Island Model + Auto-Backtracking + Novelty Search + HSS + Hybrid Trigger
# Improvements:
# 1. Re-introduced HSS (Hammersley) for diverse initialization and Resets.
# 2. Hybrid Stagnation: Vetoes mode switch if Average Fitness is still rising.
# 3. Tuned Thresholds: Patience increased to 800 gens.

import random
import time
import statistics
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import copy

try:
    import simplifier_phase14
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

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
        
        # Phase 7.1 Specifics
        self.novelty_threshold = 800   # Increased patience (was 200)
        self.backtrack_threshold = 1500 # Hard ceiling for backtracking
        self.avg_patience_factor = 0.005 # 0.5% improvement in avg required to delay switch

# --- Gates ---
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
    "NOT": (b_NOT, 1), "NAND": (b_NAND, 2), "NOR": (b_NOR, 2), "XNOR": (b_XNOR, 2), "XNOR2": (b_XNOR, 2),
    "MUX2": (b_MUX2, 3)
}
PRIMITIVES = ["AND", "OR", "XOR", "NOT", "NAND", "NOR"]
MACROS = ["MUX2"]

# --- HSS Helpers (Re-Enabled) ---
def radical_inverse(base, index):
    inv, denom = 0.0, 1.0
    while index > 0:
        index, rem = divmod(index, base)
        denom *= base
        inv += rem / denom
    return inv

def hammersley_point(i, n, dims):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] # Extended primes
    pt = [i / max(1, n)]
    for d in range(dims - 1):
        b = primes[d % len(primes)]
        pt.append(radical_inverse(b, i))
    return pt

def _hss_take(vec, idx):
    return vec[idx % len(vec)], idx + 1

# --- Helpers ---
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

# --- HSS Genome Generation ---
def random_gate_hss(idx, num_inputs, p_prim, vec, vec_idx):
    limit = num_inputs + idx
    # Type
    v, vec_idx = _hss_take(vec, vec_idx)
    if v < p_prim: gtype = random.choice(PRIMITIVES)
    else: gtype = random.choice(MACROS)
    
    _, arity = GATE_OPS[gtype]
    ins = []
    if limit > 0:
        for _ in range(arity):
            v, vec_idx = _hss_take(vec, vec_idx)
            ins.append(int(v * limit) % limit)
    else:
        ins = [0] * arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}, vec_idx

def hss_individual(num_inputs, cfg, vec, locked_prefix=None):
    """
    Generates an individual. 
    If locked_prefix is provided, copies it and fills the TAIL using HSS.
    """
    indiv = []
    vec_idx = 0
    start_idx = 0
    
    # 1. Copy Locks
    if locked_prefix:
        indiv = [g.copy() for g in locked_prefix]
        start_idx = len(locked_prefix)

    # 2. Generate Tail
    for i in range(start_idx, cfg.num_gates):
        gate, vec_idx = random_gate_hss(i, num_inputs, cfg.p_choose_primitive, vec, vec_idx)
        indiv.append(gate)
    return indiv

def init_population_hss(num_inputs, cfg, size, locked_prefix=None):
    dims = max(10, 4 * (cfg.num_gates - (len(locked_prefix) if locked_prefix else 0)))
    hss_vectors = [hammersley_point(i, size, dims) for i in range(size)]
    return [hss_individual(num_inputs, cfg, v, locked_prefix) for v in hss_vectors]

# --- Mutation (Standard) ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def mutate(ind, num_inputs, rate, cfg, locked_count):
    new_ind = []
    for i, gate in enumerate(ind):
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

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target_idx, mode="ACCURACY"):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    num_gates = len(individual)
    num_targets = len(packed_targets)
    max_score_per_col = mask.bit_count()
    
    # Identify Signal for Current Target
    if current_target_idx in solved_output_map:
        sig_idx = num_inputs + solved_output_map[current_target_idx]
    else:
        sig_idx = num_inputs + (num_gates - num_targets + current_target_idx)
    
    out_val = signals[sig_idx] if sig_idx < len(signals) else 0
    
    if mode == "NOVELTY":
        return out_val, out_val # Return signature
        
    diff = out_val ^ packed_targets[current_target_idx]
    current_score = max_score_per_col - diff.bit_count()
    
    penalty = 0
    for i in range(current_target_idx):
        if i in solved_output_map:
            s_idx = num_inputs + solved_output_map[i]
            val = signals[s_idx] if s_idx < len(signals) else 0
            if (val ^ packed_targets[i]) != 0: penalty += 50000

    return current_score - penalty, current_score

# --- Parallel ---
_PE_data = {}
def _init_pool(inputs, targets, mask, n_in, solved_map, curr_target, mode):
    _PE_data['inputs'] = inputs
    _PE_data['targets'] = targets
    _PE_data['mask'] = mask
    _PE_data['n_in'] = n_in
    _PE_data['solved_map'] = solved_map
    _PE_data['curr_target'] = curr_target
    _PE_data['mode'] = mode

def _eval_wrapper(ind):
    return fitness_bitwise(ind, _PE_data['inputs'], _PE_data['targets'], _PE_data['mask'], 
                           _PE_data['n_in'], _PE_data['solved_map'], _PE_data['curr_target'], _PE_data['mode'])

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
    for out_i in solved_indices:
        if out_i in solved_output_map: gate_idx = solved_output_map[out_i]
        else: gate_idx = len(best_ind) - num_outputs + out_i
        
        stack = [gate_idx]
        keep_indices.add(gate_idx)
        while stack:
            curr = stack.pop()
            if curr >= len(best_ind): continue
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
        new_inputs = [num_inputs + remap.get(inp - num_inputs, 0) if inp >= num_inputs else inp for inp in gate['inputs']]
        gate['inputs'] = new_inputs
        gate['name'] = f"G{len(new_genome)}"
        new_genome.append(gate)
        
    locked_count = len(new_genome)
    final_output_map = {}
    for out_i in solved_indices:
        old = solved_output_map.get(out_i, len(best_ind) - num_outputs + out_i)
        final_output_map[out_i] = remap.get(old, 0)
        
    return new_genome, locked_count, final_output_map

# --- Main Island Loop ---
def evolve_phase71(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    num_islands = cfg.num_islands
    island_pop_size = cfg.pop_size // num_islands
    print(f"🏝️ Phase 7.1: {num_islands} Islands. Hybrid Stagnation + HSS Reset.")
    
    # HSS Init
    islands = [init_population_hss(num_inputs, cfg, island_pop_size) for _ in range(num_islands)]
    
    solved_mask = [False] * num_outputs
    solved_output_map = {}
    locked_count = 0
    current_target = 0
    
    stagnation_counter = 0
    best_score_history = -1
    best_avg_history = -1 # New: Track Average for Hybrid Trigger
    evolution_mode = "ACCURACY"
    
    checkpoints = {} 
    
    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))

    try:
        for gen in range(cfg.generations):
            
            flat_pop = [ind for island in islands for ind in island]
            results = pool.map(_eval_wrapper, flat_pop)
            
            offset = 0
            global_best_raw = -1
            global_best_ind = None
            global_mu = 0 # Track global average
            
            new_islands = []
            
            for i in range(num_islands):
                res_slice = results[offset : offset + island_pop_size]
                island_pop = islands[i]
                offset += island_pop_size
                
                if evolution_mode == "ACCURACY":
                    scalars = [r[0] for r in res_slice]
                    raws = [r[1] for r in res_slice]
                    global_mu += sum(raws) # Accumulate raw scores
                else: 
                    sigs = [r[0] for r in res_slice]
                    scalars = []
                    for sig in sigs:
                        dist = 0
                        for _ in range(5):
                            peer = random.choice(sigs)
                            dist += (sig ^ peer).bit_count()
                        scalars.append(dist)
                    raws = [0] * len(island_pop)
                
                # Global Best & Mode specific logic
                if evolution_mode == "NOVELTY":
                    # --- PEEKING LOGIC (Preserved) ---
                    local_best_dist = max(scalars)
                    local_idx = scalars.index(local_best_dist)
                    novel_ind = island_pop[local_idx]
                    
                    test_signals = evaluate_bitwise(novel_ind, packed_inputs, mask, num_inputs)
                    
                    if current_target in solved_output_map:
                        sig_idx = num_inputs + solved_output_map[current_target]
                    else:
                        sig_idx = num_inputs + (len(novel_ind) - len(packed_targets) + current_target)
                    
                    out_val = test_signals[sig_idx] if sig_idx < len(test_signals) else 0
                    diff = out_val ^ packed_targets[current_target]
                    true_score = max_score_per_col - diff.bit_count()
                    
                    if true_score > global_best_raw:
                        global_best_raw = true_score
                        global_best_ind = novel_ind
                else:
                    # Accuracy Mode
                    local_best = max(scalars)
                    local_idx = scalars.index(local_best)
                    if raws[local_idx] > global_best_raw:
                        global_best_raw = raws[local_idx]
                        global_best_ind = island_pop[local_idx]

                # Evolution
                new_pop = []
                sorted_pop = [x for _, x in sorted(zip(scalars, island_pop), key=lambda p: p[0], reverse=True)]
                new_pop.extend([g[:] for g in sorted_pop[:cfg.elitism]])
                
                while len(new_pop) < island_pop_size:
                    p1 = random.choice(island_pop)
                    p2 = random.choice(island_pop)
                    c1, c2 = crossover(p1, p2)
                    new_pop.append(mutate(c1, num_inputs, cfg.base_mut, cfg, locked_count))
                    if len(new_pop) < island_pop_size:
                        new_pop.append(mutate(c2, num_inputs, cfg.base_mut, cfg, locked_count))
                new_islands.append(new_pop)
            
            islands = new_islands
            global_mu /= (num_islands * island_pop_size) # Normalize average

            # --- Logic Control ---
            
            # A. Success
            if global_best_raw == max_score_per_col:
                print(f"🎉 Output #{current_target+1} SOLVED at Gen {gen}!")
                
                solved_mask[current_target] = True
                indices = [k for k, x in enumerate(solved_mask) if x]
                l_genome, l_count, l_map = defragment_genome(global_best_ind, indices, solved_output_map, num_inputs, cfg.num_gates, num_outputs)
                
                checkpoints[current_target] = (l_genome, l_count, l_map)
                current_target += 1
                solved_output_map = l_map
                locked_count = l_count
                
                stagnation_counter = 0
                best_score_history = -1
                best_avg_history = -1
                evolution_mode = "ACCURACY"
                
                if current_target >= num_outputs:
                    print("✅ ALL OUTPUTS SOLVED!")
                    return global_best_ind
                
                print(f"   >> Advancing to Target #{current_target+1}. Locked Gates: {locked_count}")
                
                pool.close(); pool.join()
                pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                            initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
                
                # --- HSS RESET ---
                # Re-populate using HSS for the tails
                print("   >> 🌊 HSS Global Population Reset...")
                islands = [init_population_hss(num_inputs, cfg, island_pop_size, locked_prefix=l_genome) for _ in range(num_islands)]
                continue

            # B. Hybrid Stagnation
            if global_best_raw > best_score_history:
                best_score_history = global_best_raw
                stagnation_counter = 0
                # Also reset avg tracking on new best
                best_avg_history = global_mu
                
                if evolution_mode == "NOVELTY":
                    print("   >> Novelty found better accuracy! Switching back to ACCURACY.")
                    evolution_mode = "ACCURACY"
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
            else:
                stagnation_counter += 1
                # Track best average seen during this plateau
                if global_mu > best_avg_history:
                    best_avg_history = global_mu
            
            # C. Trigger Backtrack
            if stagnation_counter > cfg.backtrack_threshold and current_target > 0:
                print(f"⚠️ HARD STAGNATION. BACKTRACKING.")
                revert_idx = max(0, current_target - 1)
                
                if revert_idx == 0: l_genome, l_count, l_map = [], 0, {}
                else: l_genome, l_count, l_map = checkpoints[revert_idx-1]
                    
                current_target = revert_idx
                solved_output_map = l_map
                locked_count = l_count
                solved_mask[current_target] = False
                
                # HSS Reset on Backtrack too
                islands = [init_population_hss(num_inputs, cfg, island_pop_size, locked_prefix=l_genome) for _ in range(num_islands)]
                        
                stagnation_counter = 0
                best_score_history = -1
                evolution_mode = "ACCURACY"
                
                pool.close(); pool.join()
                pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                            initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
                continue

            # D. Trigger Novelty (Hybrid Veto)
            if stagnation_counter > cfg.novelty_threshold and evolution_mode == "ACCURACY":
                # Hybrid Veto: Is average still rising significantly?
                # We compare current mu to the best mu seen at start of plateau?
                # Simplified: If mu is > 99.5% of best_avg_history, we assume we saturated.
                # If mu is rapidly rising, it will constantly update best_avg_history.
                # So we really want to know: "Has average improved recently?"
                # Actually, the user logic was: VETO if rising.
                # So if we are improving, we deduct stagnation.
                
                # Implementation: If we hit threshold, but Average is High relative to history, extend.
                # Actually, let's just check if we updated best_avg_history recently? 
                # No, let's stick to the requested logic:
                
                # If current global_mu is notably higher than what it was 50 gens ago?
                # Too complex to track 50 gens ago. 
                # Let's just say: If global_mu is continuously breaking records, we reset the counter slightly.
                # Actually, let's implement the 'Simple' Veto:
                # We effectively use 'best_avg_history' as a secondary best score.
                
                if global_mu >= best_avg_history * (1.0 + cfg.avg_patience_factor):
                    # Average has improved by > 0.5% since the last 'Best Score' update
                    # This means the population is learning even if the leader is stuck.
                    print(f"   >> VETO: Best stuck, but Avg rising ({global_mu:.1f}). Buying time.")
                    stagnation_counter = int(cfg.novelty_threshold * 0.8) # Knock back 20%
                    best_avg_history = global_mu # Update benchmark
                else:
                    print(f"⚠️ Stagnation ({stagnation_counter} gens). Switching to NOVELTY SEARCH.")
                    evolution_mode = "NOVELTY"
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "NOVELTY"))

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Target #{current_target+1} | Score: {global_best_raw}/{max_score_per_col} | Avg: {global_mu:.1f}")

    finally:
        if pool: pool.close(); pool.join()

    return global_best_ind