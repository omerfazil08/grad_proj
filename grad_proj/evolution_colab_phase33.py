# evolution_colab_phase32.py
# Phase 3.2: Bitwise Engine + Constructive Locking (The "Hardware Freeze" Update)
# Implements user suggestion: Trace paths of solved outputs and lock specific gates.

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

# --- HSS Helpers ---
def hammersley_point(i, n, dims):
    def radical_inverse(base, index):
        inv, denom = 0.0, 1.0
        while index > 0:
            index, rem = divmod(index, base)
            denom *= base
            inv += rem / denom
        return inv
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    pt = [i / max(1, n)]
    for d in range(dims - 1):
        b = primes[d % len(primes)]
        pt.append(radical_inverse(b, i))
    return pt

def _hss_take(vec, idx):
    return vec[idx % len(vec)], idx + 1

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

# --- Path Tracing for Locking ---
def trace_active_gates(individual, output_idx, num_inputs, num_outputs):
    """
    Backtracks from the specified output gate to find all gate indices 
    that contribute to this output's logic.
    """
    active_indices = set()
    
    # Output gates are the last N gates
    start_gate_idx = len(individual) - num_outputs + output_idx
    
    # Check if circuit is valid size
    if start_gate_idx < 0: return set()
    
    stack = [start_gate_idx]
    while stack:
        idx = stack.pop()
        if idx in active_indices: continue
        
        # Valid gate index?
        if 0 <= idx < len(individual):
            active_indices.add(idx)
            gate = individual[idx]
            
            # Check inputs
            for inp in gate['inputs']:
                # Inputs < num_inputs are primary inputs (ignore)
                # Inputs >= num_inputs are gate references
                if inp >= num_inputs:
                    # Convert input ref back to gate index
                    # gate ref 10 (if num_inputs=3) -> Gate index 7
                    source_gate_idx = inp - num_inputs
                    stack.append(source_gate_idx)
                    
    return active_indices

# --- Genome Gen ---
def random_gate_hss(idx, num_inputs, p_prim, vec, vec_idx):
    limit = num_inputs + idx
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

def hss_individual(num_inputs, cfg, vec):
    indiv = []
    vec_idx = 0
    for i in range(cfg.num_gates):
        gate, vec_idx = random_gate_hss(i, num_inputs, cfg.p_choose_primitive, vec, vec_idx)
        indiv.append(gate)
    return indiv

def init_population(num_inputs, cfg):
    dims = max(10, 4 * cfg.num_gates)
    hss_vectors = [hammersley_point(i, cfg.pop_size, dims) for i in range(cfg.pop_size)]
    return [hss_individual(num_inputs, cfg, v) for v in hss_vectors]

# --- Mutation (Aware of Locks) ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    if limit > 0:
        ins = [random.randint(0, limit - 1) for _ in range(arity)]
    else:
        ins = [0] * arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def mutate(ind, num_inputs, rate, cfg, locked_gates):
    """
    Mutates the individual, but PRESERVES any gate index found in locked_gates.
    """
    new_ind = []
    for i, gate in enumerate(ind):
        # If locked, COPY EXACTLY (Do not touch)
        if i in locked_gates:
            new_ind.append(gate) # Reference copy is fine as we won't modify it
            continue
            
        if random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

# --- Evaluation ---
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

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    output_signals = signals[-len(packed_targets):]
    scores = []
    for i, target in enumerate(packed_targets):
        if i >= len(output_signals): out_val = 0
        else: out_val = output_signals[i]
        diff = out_val ^ target
        scores.append(mask.bit_count() - diff.bit_count())
    return scores

# --- Parallel Wrapper ---
_PE_data = {}
def _init_pool(inputs, targets, mask, n_in):
    _PE_data['inputs'] = inputs
    _PE_data['targets'] = targets
    _PE_data['mask'] = mask
    _PE_data['n_in'] = n_in

def _eval_wrapper(ind):
    scores = fitness_bitwise(ind, _PE_data['inputs'], _PE_data['targets'], _PE_data['mask'], _PE_data['n_in'])
    # Scalar is just sum, no penalties needed as structure enforces locks
    return sum(scores), scores

# --- Main Evolution ---
def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def get_genome_hash(ind):
    return "".join([f"{g['type']}{g['inputs']}" for g in ind])

def evolve_bitwise(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    print(f"Bitwise v2.5 (Phase 3.2): Hardware Locking Enabled.")
    
    population = init_population(num_inputs, cfg)
    solved_mask = [False] * num_outputs
    
    # LOCKED GATES DATABASE: {gate_index: gate_dict}
    # These gates are "frozen hardware" and will be forced into every individual
    locked_gates = {} 
    
    history = {'gen': [], 'best': [], 'mu': [], 'sigma': []}
    
    # Stagnation tracking on AVERAGE
    last_avg_improvement = 0
    best_avg_so_far = -float('inf')
    
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_targets, mask, num_inputs))

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
                    sc = fitness_bitwise(ind, packed_inputs, packed_targets, mask, num_inputs)
                    scalars.append(sum(sc))
                    breakdowns.append(sc)
            
            # Stats
            best_val = max(scalars)
            best_idx = scalars.index(best_val)
            best_bkd = breakdowns[best_idx]
            mu = statistics.mean(scalars)
            sigma = statistics.stdev(scalars) if len(scalars) > 1 else 0
            
            history['gen'].append(gen)
            history['best'].append(best_val)
            history['mu'].append(mu)
            history['sigma'].append(sigma)

            # Check Solved Outputs
            new_solve = False
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    
                    # --- CRITICAL: LOCK THE LOGIC ---
                    # 1. Identify which gates created this solution
                    active_indices = trace_active_gates(population[best_idx], i, num_inputs, num_outputs)
                    
                    # 2. Add to lock list
                    print(f"   >> Locking {len(active_indices)} gates for Output {i+1}")
                    for idx in active_indices:
                        locked_gates[idx] = population[best_idx][idx]
                        
                    new_solve = True

            # Apply Locks (Population Synchronization)
            # If we have new locks, we force them onto the population immediately
            # This ensures NO child loses the progress.
            if new_solve or (gen == 0): # Apply on gen 0 too just in case
                for i in range(len(population)):
                    for idx, gate in locked_gates.items():
                        population[i][idx] = gate # Overwrite with locked logic
                
                # Reset stats because we artificially boosted the population
                best_avg_so_far = -float('inf')
                last_avg_improvement = gen

            # Stagnation Logic (Average Fitness)
            if mu > best_avg_so_far:
                best_avg_so_far = mu
                last_avg_improvement = gen
            
            is_stagnant = (gen - last_avg_improvement) > 150
            current_mut_rate = cfg.base_mut * (1 - gen/cfg.generations) + cfg.min_mut
            
            if is_stagnant:
                current_mut_rate = 0.40
                if gen % 10 == 0: 
                    print(f"⚠️ Avg Fitness Stagnated (Last impr: Gen {last_avg_improvement}). Hyper-Mutation.")

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | Avg={mu:.1f} | {best_bkd}")
                
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[best_idx], best_bkd, {}, history

            # Reproduction
            new_pop = []
            sorted_indices = sorted(range(len(scalars)), key=lambda k: scalars[k], reverse=True)
            
            # Unique Elitism
            seen_hashes = set()
            elites_added = 0
            for idx in sorted_indices:
                ind = population[idx]
                h = get_genome_hash(ind)
                if h not in seen_hashes:
                    new_pop.append(ind)
                    seen_hashes.add(h)
                    elites_added += 1
                if elites_added >= cfg.elitism: break
            
            while len(new_pop) < cfg.pop_size:
                t1 = random.sample(range(len(population)), cfg.tournament_k)
                p1 = population[max(t1, key=lambda i: scalars[i])]
                t2 = random.sample(range(len(population)), cfg.tournament_k)
                p2 = population[max(t2, key=lambda i: scalars[i])]
                
                c1, c2 = crossover(p1, p2)
                
                # Mutate (passing locked_gates to protect them)
                new_pop.append(mutate(c1, num_inputs, current_mut_rate, cfg, locked_gates))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, current_mut_rate, cfg, locked_gates))
            
            # Re-apply locks to children (Double safety)
            for i in range(len(new_pop)):
                 for idx, gate in locked_gates.items():
                    new_pop[i][idx] = gate

            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    return population[best_idx], best_bkd, {}, history

def convert_to_string_format(individual, num_inputs):
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