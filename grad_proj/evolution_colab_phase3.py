# evolution_bitwise.py
# Phase 3: Bitwise Parallel Evolution Engine
# Optimizes fitness evaluation by packing truth tables into integers.
# 100x-1000x faster than row-by-row evaluation for large tables.

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
# These operate on full Integers (Truth Vectors) rather than single bits.
# 'mask' is required for NOT/NAND/NOR to prevent infinite 1s in Python negative ints.

def b_AND(a, b, mask): return a & b
def b_OR(a, b, mask):  return a | b
def b_XOR(a, b, mask): return a ^ b
def b_NOT(a, mask):    return (~a) & mask
def b_NAND(a, b, mask): return (~(a & b)) & mask
def b_NOR(a, b, mask):  return (~(a | b)) & mask
def b_XNOR(a, b, mask): return (~(a ^ b)) & mask

# Macros (Optimized for bitwise)
def b_MUX2(s, a, b, mask): 
    # if s=0 choose a, else choose b -> (~s & a) | (s & b)
    return (((~s) & mask) & a) | (s & b)

# Gate Registry (Function, Arity)
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
    """
    Converts list of tuples [(0,0), (0,1)...] into list of Integers.
    Returns: (packed_inputs, mask)
    """
    num_rows = len(inputs)
    mask = (1 << num_rows) - 1
    packed = [0] * num_inputs
    
    for row_idx, row in enumerate(inputs):
        for col_idx, bit in enumerate(row):
            if bit:
                packed[col_idx] |= (1 << row_idx)
    return packed, mask

def pack_targets(targets):
    """
    Converts target columns [[0,1...], [1,1...]] into Integers.
    """
    packed = []
    for col in targets:
        val = 0
        for r, bit in enumerate(col):
            if bit:
                val |= (1 << r)
        packed.append(val)
    return packed

# --- Genome & Init ---
def random_gate(idx, num_inputs, p_prim):
    # Determine allowed inputs
    # Input indices: 0..(num_inputs-1) are primary inputs
    # Input indices: num_inputs..(num_inputs+idx-1) are previous gates
    limit = num_inputs + idx
    
    # Choose Type
    if random.random() < p_prim:
        gtype = random.choice(PRIMITIVES)
    else:
        gtype = random.choice(MACROS)
        
    func, arity = GATE_OPS[gtype]
    
    # Choose Inputs (Integers representing indices)
    if limit == 0: # Edge case for first gate with 0 inputs? Should not happen if num_inputs >= 1
        ins = [0] * arity 
    else:
        ins = [random.randint(0, limit - 1) for _ in range(arity)]
        
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def random_individual(num_inputs, cfg):
    return [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]

def init_population(num_inputs, cfg):
    # HSS omitted for brevity in bitwise engine, random is sufficient for large scale usually,
    # but we can re-add if needed. Random is faster to generate.
    return [random_individual(num_inputs, cfg) for _ in range(cfg.pop_size)]

# --- Evaluation (The Fast Part) ---
def evaluate_bitwise(individual, packed_inputs, mask, num_inputs):
    # Array to store the 'Truth Integer' of every signal
    # Indices 0..num_inputs-1 are Primary Inputs
    # Indices num_inputs..end are Gate Outputs
    
    # Pre-fill with inputs
    signals = list(packed_inputs) 
    
    for gate in individual:
        gtype = gate['type']
        ins = gate['inputs'] # These are integer indices into 'signals' array
        
        func, arity = GATE_OPS[gtype]
        
        # Fetch operand values
        vals = [signals[i] if i < len(signals) else 0 for i in ins]
        
        # Compute
        if arity == 1:
            res = func(vals[0], mask)
        elif arity == 2:
            res = func(vals[0], vals[1], mask)
        elif arity == 3:
            res = func(vals[0], vals[1], vals[2], mask)
            
        signals.append(res)
        
    return signals

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs):
    # Eval
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    
    # Outputs are the last N signals
    num_outputs = len(packed_targets)
    output_signals = signals[-num_outputs:]
    
    scores = []
    for i, target in enumerate(packed_targets):
        # XOR gives bits that are DIFFERENT. 
        # NOT XOR (XNOR) gives bits that are SAME.
        # We want matches.
        # But wait, easier: XOR gives differences.
        # Hamming Distance = popcount(out ^ target)
        # Matches = Total_Rows - Hamming_Distance
        
        if i >= len(output_signals): # Circuit too small?
            out_val = 0
        else:
            out_val = output_signals[i]
            
        diff = out_val ^ target
        # Count 1s (errors)
        errors = diff.bit_count() # Python 3.10+
        matches = mask.bit_count() - errors
        scores.append(matches)
        
    return scores

# --- Parallel Wrapper ---
_PE_inputs = None
_PE_targets = None
_PE_mask = None
_PE_n_in = None
_PE_solved_mask = None
_PE_max_score = 0

def _init_pool(inputs, targets, mask, n_in, solved):
    global _PE_inputs, _PE_targets, _PE_mask, _PE_n_in, _PE_solved_mask, _PE_max_score
    _PE_inputs = inputs
    _PE_targets = targets
    _PE_mask = mask
    _PE_n_in = n_in
    _PE_solved_mask = solved
    _PE_max_score = mask.bit_count()

def _eval_wrapper(ind):
    scores = fitness_bitwise(ind, _PE_inputs, _PE_targets, _PE_mask, _PE_n_in)
    
    # Calculate scalar fitness with penalty
    scalar = sum(scores)
    for i, s in enumerate(scores):
        if _PE_solved_mask[i] and s < _PE_max_score:
            scalar -= 10000 # Huge penalty
            
    return scalar, scores

# --- Evolution Loop ---
def mutate(ind, num_inputs, rate, cfg):
    new_ind = []
    limit = num_inputs
    for i, gate in enumerate(ind):
        if random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
        limit += 1
    return new_ind

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def evolve_bitwise(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    
    # 1. Convert Truth Tables to Integers (Bit Packing)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count() # Should be 2^N
    total_max = max_score_per_col * num_outputs
    
    print(f"Bitwise Mode: Compressed {len(inputs_list)} rows into {mask.bit_length()}-bit integers.")
    
    # 2. Init Pop
    population = init_population(num_inputs, cfg)
    
    solved_mask = [False] * num_outputs
    hall_of_fame = {}
    history = {'gen': [], 'best': [], 'mu': [], 'sigma': []}
    
    # Pool
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_mask))

    try:
        for gen in range(cfg.generations):
            # Eval
            if pool:
                results = pool.map(_eval_wrapper, population)
                scalars = [r[0] for r in results]
                breakdowns = [r[1] for r in results]
            else:
                # Serial fallback
                scalars = []
                breakdowns = []
                for ind in population:
                    sc = fitness_bitwise(ind, packed_inputs, packed_targets, mask, num_inputs)
                    s_val = sum(sc)
                    for i, s in enumerate(sc):
                        if solved_mask[i] and s < max_score_per_col: s_val -= 10000
                    scalars.append(s_val)
                    breakdowns.append(sc)
            
            # Stats
            best_val = max(scalars)
            best_idx = scalars.index(best_val)
            best_bkd = breakdowns[best_idx]
            
            mu = statistics.mean(scalars)
            sigma = statistics.stdev(scalars) if len(scalars)>1 else 0
            
            history['gen'].append(gen)
            history['best'].append(best_val)
            history['mu'].append(mu)
            history['sigma'].append(sigma)
            
            # Check Solved
            new_solve = False
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    hall_of_fame[i] = population[best_idx]
                    new_solve = True
            
            if new_solve and pool:
                # Update pool globals with new solved mask
                pool.close(); pool.join()
                pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                            initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_mask))

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | {best_bkd}")
                
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[best_idx], best_bkd, hall_of_fame, history

            # Reproduction
            new_pop = []
            # Elitism
            sorted_pop = [x for _, x in sorted(zip(scalars, population), key=lambda pair: pair[0], reverse=True)]
            new_pop.extend(sorted_pop[:cfg.elitism])
            
            # HOF
            for k, v in hall_of_fame.items():
                new_pop.append(v)
                
            # Fill
            while len(new_pop) < cfg.pop_size:
                # Tournament
                t_idx = random.sample(range(len(population)), cfg.tournament_k)
                p1 = population[max(t_idx, key=lambda i: scalars[i])]
                t_idx = random.sample(range(len(population)), cfg.tournament_k)
                p2 = population[max(t_idx, key=lambda i: scalars[i])]
                
                c1, c2 = crossover(p1, p2)
                
                # Dynamic mutation
                rate = cfg.base_mut * (1 - gen/cfg.generations) + cfg.min_mut
                new_pop.append(mutate(c1, num_inputs, rate, cfg))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, rate, cfg))
            
            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    # Return best of last gen
    return sorted_pop[0], best_bkd, hall_of_fame, history

# --- Reconstruct for Export ---
# The bitwise engine uses integer inputs. The VHDL generator expects "string" names in the gate inputs.
# We need to convert the bitwise genome (ints) back to string format before returning.

def convert_to_string_format(individual, num_inputs):
    """
    Converts internal integer indices (0..N) to strings ("A0", "G5").
    """
    string_gates = []
    for i, gate in enumerate(individual):
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs:
                str_inputs.append(f"A{inp_idx}")
            else:
                # Gate indices start after inputs. 
                # e.g. if num_inputs=3, index 3 is G0.
                gate_num = inp_idx - num_inputs
                str_inputs.append(f"G{gate_num}")
        
        string_gates.append({
            'name': f"G{i}",
            'type': gate['type'],
            'inputs': str_inputs,
            'output': f"G{i}"
        })
    return string_gates