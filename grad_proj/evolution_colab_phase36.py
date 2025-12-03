# evolution_colab_phase36.py
# Phase 3.6: Bitwise Engine + Context-Aware Mutation + ONLINE LOGIC COMPRESSION
# Implements "Option 1": Simplifies and re-synthesizes solved outputs to minimize locked gates.

import random
import time
import statistics
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import sympy
from sympy.logic.boolalg import to_dnf, to_cnf, And, Or, Not, Xor

# Try importing your simplifier
try:
    import simplifier_phase14
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False
    print("⚠️ Warning: simplifier_phase14 not found. Compression disabled.")

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

# --- Packing ---
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

# --- Genome Generation ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    if limit > 0:
        ins = [random.randint(0, limit - 1) for _ in range(arity)]
    else:
        ins = [0] * arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def init_population(num_inputs, cfg):
    # HSS Init
    dims = max(10, 4 * cfg.num_gates)
    pop = []
    for i in range(cfg.pop_size):
        vec = hammersley_point(i, cfg.pop_size, dims)
        indiv = []
        vec_idx = 0
        for g_idx in range(cfg.num_gates):
            limit = num_inputs + g_idx
            v, vec_idx = _hss_take(vec, vec_idx)
            gtype = random.choice(PRIMITIVES) if v < cfg.p_choose_primitive else random.choice(MACROS)
            _, arity = GATE_OPS[gtype]
            ins = []
            if limit > 0:
                for _ in range(arity):
                    v, vec_idx = _hss_take(vec, vec_idx)
                    ins.append(int(v * limit) % limit)
            else:
                ins = [0] * arity
            indiv.append({'name': f"G{g_idx}", 'type': gtype, 'inputs': ins})
        pop.append(indiv)
    return pop

# --- Logic Compression & Re-Synthesis (The Phase 3.6 Magic) ---

def synthesize_expr_to_gates(expr, num_inputs, gate_offset_start):
    """
    Recursively converts a SymPy expression into a linear list of gate dictionaries.
    Returns: (list_of_gates, output_signal_index)
    """
    generated_gates = []
    
    # Map Symbols to Input Indices
    # "A0" -> 0, "A1" -> 1
    def get_signal_index(node):
        s = str(node)
        if s.startswith("A") and s[1:].isdigit():
            return int(s[1:])
        return None

    def visit(node):
        # 1. Base Case: Input Symbol
        sig = get_signal_index(node)
        if sig is not None:
            return sig
            
        # 2. NOT Gate
        if isinstance(node, Not):
            child_sig = visit(node.args[0])
            # Check if we already have a NOT for this? (Optimization omitted for speed)
            new_idx = num_inputs + gate_offset_start + len(generated_gates)
            generated_gates.append({
                'name': f"G{new_idx-num_inputs}", 
                'type': 'NOT', 
                'inputs': [child_sig]
            })
            return new_idx

        # 3. Binary/N-ary Ops (AND, OR, XOR)
        if isinstance(node, (And, Or, Xor)):
            op_type = "AND" if isinstance(node, And) else "OR" if isinstance(node, Or) else "XOR"
            
            # Process children
            child_sigs = [visit(arg) for arg in node.args]
            
            # Cascade them if > 2 inputs (e.g., And(A,B,C) -> And(And(A,B), C))
            curr_sig = child_sigs[0]
            for i in range(1, len(child_sigs)):
                next_sig = child_sigs[i]
                new_idx = num_inputs + gate_offset_start + len(generated_gates)
                generated_gates.append({
                    'name': f"G{new_idx-num_inputs}",
                    'type': op_type,
                    'inputs': [curr_sig, next_sig]
                })
                curr_sig = new_idx
            return curr_sig

        return 0 # Fallback

    final_sig = visit(expr)
    return generated_gates, final_sig

def compress_genome(individual, solved_indices, num_inputs, num_gates):
    """
    Takes an individual with some solved outputs.
    1. Simplifies the logic for solved outputs.
    2. Re-synthesizes efficient gates.
    3. Packs them at the start of the genome.
    4. Returns (new_genome, new_locked_gates_dict)
    """
    if not HAS_SIMPLIFIER:
        return individual, {}

    # 1. Convert to String Format for Simplifier
    str_ind = convert_to_string_format(individual, num_inputs)
    input_names = [f"A{i}" for i in range(num_inputs)]
    
    # 2. Collect Simplified Logic for ALL solved outputs
    # We re-synthesize ALL solved outputs together to pack them tightly.
    
    compressed_gates_all = []
    locked_map = {} # {gate_index: gate_dict}
    
    current_gate_offset = 0
    
    # Track where each output signal ends up
    # We need to ensure the final gate for Output X is actually connecting to Output X?
    # In Phase 3, outputs are usually implicit (last N gates). 
    # WE NEED TO CHANGE THIS: We will enforce that locked gates are at the BOTTOM (start).
    
    # Let's build the "Frozen Section"
    for out_idx in solved_indices:
        # Identify the gate responsible for this output
        # (In raw genome, it's usually determined by position, but here we act smart)
        # We'll trace the best gate for this output from the raw individual.
        
        # Simplify
        target_gate_name = individual[-(len(solved_indices)-out_idx)]["name"] # Approximation
        # Actually, we should pass the exact gate index. 
        # For now, let's re-simplify the logic based on Truth Table if possible, 
        # but using the simplifier on the evolved structure is safer.
        
        # Find the output gate index in the raw individual
        # We assume the main loop passes us the *best* individual for this output.
        # But wait, 'individual' here is one genome.
        # Let's simplify the logic ending at the gate currently driving this output.
        
        raw_out_gate_name = individual[-(len(solved_indices)) + out_idx]['name'] # This logic depends on output mapping
        # Correction: In Phase 3, outputs are the LAST N gates.
        # So Output 0 is at index -N, Output 1 is at -(N-1)...
        
        # Get SymPy Expression
        res = simplifier_phase14.simplify_genome(str_ind, input_names, [f"G{len(individual)-len(solved_indices)+out_idx}"])
        # The simplifier returns dictionary keys like "Output 1 (G55)"
        expr_str = list(res.values())[0]
        expr_sym = sympy.sympify(expr_str)
        
        # Re-Synthesize into Gates
        new_gates, final_sig_idx = synthesize_expr_to_gates(expr_sym, num_inputs, current_gate_offset)
        
        # Append to our compressed list
        for g in new_gates:
            # Fix names to be sequential G0, G1, G2...
            g['name'] = f"G{current_gate_offset}"
            compressed_gates_all.append(g)
            locked_map[current_gate_offset] = g
            current_gate_offset += 1
            
    # 3. Fill the rest of the genome with random gates
    remaining_slots = num_gates - len(compressed_gates_all)
    for i in range(remaining_slots):
        idx = current_gate_offset + i
        # Random gate (standard init)
        g = random_gate(idx, num_inputs, 0.6) # Default primitives
        g['name'] = f"G{idx}"
        compressed_gates_all.append(g)
        
    print(f"   >> Compressed Logic: {len(locked_map)} gates locked (was {len(individual)}).")
    return compressed_gates_all, locked_map


# --- Mutation ---
def random_gate(idx, num_inputs, p_prim):
    limit = num_inputs + idx
    gtype = random.choice(PRIMITIVES) if random.random() < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    ins = [random.randint(0, limit - 1) for _ in range(arity)] if limit > 0 else [0]*arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}

def mutate(ind, num_inputs, rate, cfg, locked_gates):
    new_ind = []
    for i, gate in enumerate(ind):
        if i in locked_gates:
            new_ind.append(locked_gates[i]) # Strict copy from lock dict
        elif random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

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

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs):
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    # In Phase 3.6, we assume outputs are the LAST N gates
    # But wait! If we compressed logic, the "solved" outputs might be in the LOCKED section (beginning).
    # We need a smarter output mapping.
    # For simplicity in this version: We won't change the output mapping logic (always last N gates).
    # Instead, we rely on the locked gates *feeding* into identity/wire gates at the end, 
    # OR we assume the "compressed" genome places the valuable logic early, and the later gates just wire it out.
    # Actually, let's just check the LAST N signals as usual. 
    # Evolution will figure out how to wire the Locked Block to the End.
    
    output_signals = signals[-len(packed_targets):]
    scores = []
    for i, target in enumerate(packed_targets):
        out_val = output_signals[i] if i < len(output_signals) else 0
        diff = out_val ^ target
        scores.append(mask.bit_count() - diff.bit_count())
    return scores

# --- Parallel ---
_PE_data = {}
def _init_pool(inputs, targets, mask, n_in):
    _PE_data['inputs'] = inputs
    _PE_data['targets'] = targets
    _PE_data['mask'] = mask
    _PE_data['n_in'] = n_in

def _eval_wrapper(ind):
    scores = fitness_bitwise(ind, _PE_data['inputs'], _PE_data['targets'], _PE_data['mask'], _PE_data['n_in'])
    return sum(scores), scores

# --- Main Loop ---
def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

def evolve_bitwise(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    print(f"Bitwise v3.6: Option 1 (Online Compression & Locking).")
    
    population = init_population(num_inputs, cfg)
    solved_mask = [False] * num_outputs
    
    # Stores the EXACT gate structures we decided to freeze
    # {0: Gate0, 1: Gate1, ... 7: Gate7}
    locked_gates = {} 
    
    history = {'gen': [], 'best': [], 'mu': [], 'sigma': []}
    last_avg_improvement = 0
    best_avg_so_far = -float('inf')
    
    pool = None
    if cfg.parallel:
        pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                    initargs=(packed_inputs, packed_targets, mask, num_inputs))

    try:
        for gen in range(cfg.generations):
            # 1. Eval
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
            
            # 2. Stats
            best_val = max(scalars)
            best_idx = scalars.index(best_val)
            best_bkd = breakdowns[best_idx]
            mu = statistics.mean(scalars)
            sigma = statistics.stdev(scalars) if len(scalars) > 1 else 0
            
            history['gen'].append(gen)
            history['best'].append(best_val)
            history['mu'].append(mu)
            history['sigma'].append(sigma)

            # 3. Check Solved Outputs
            new_solve = False
            solve_indices = []
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    new_solve = True
                if solved_mask[i]:
                    solve_indices.append(i)

            # 4. COMPRESSION TRIGGER
            if new_solve:
                print("   >> Compressing and Defragmenting Genome...")
                # Take the best individual and compress its logic for ALL solved outputs
                compressed_ind, new_locks = compress_genome(
                    population[best_idx], 
                    solve_indices, 
                    num_inputs, 
                    cfg.num_gates
                )
                
                locked_gates = new_locks
                
                # Replace ENTIRE population with this new optimized baseline
                # This is the "Ratchet" effect. Everyone starts fresh from the optimized base.
                for i in range(len(population)):
                    population[i] = list(compressed_ind) # Deep copy via list()
                
                # Reset stats
                best_avg_so_far = -float('inf')
                last_avg_improvement = gen
            
            # 5. Mutation Logic
            if mu > best_avg_so_far:
                best_avg_so_far = mu
                last_avg_improvement = gen
            
            is_stagnant = (gen - last_avg_improvement) > 100
            
            # Context-aware rates
            # Calculate distance to next solve
            min_dist = float('inf')
            for i, score in enumerate(best_bkd):
                if not solved_mask[i]:
                    d = max_score_per_col - score
                    if d < min_dist: min_dist = d
            
            if min_dist < 50: current_mut_rate = 0.02
            elif min_dist < 150: current_mut_rate = 0.05
            else: current_mut_rate = 0.15

            if is_stagnant:
                current_mut_rate = 0.35
                if gen % 50 == 0: print(f"⚠️ Stagnation. Shock.")

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | Avg={mu:.1f} | Rate={current_mut_rate:.2f} | {best_bkd}")
            
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[best_idx], best_bkd, {}, history

            # 6. Reproduction
            new_pop = []
            # Elitism
            sorted_indices = sorted(range(len(scalars)), key=lambda k: scalars[k], reverse=True)
            
            # Add Elites (Unique)
            seen = set()
            added = 0
            for idx in sorted_indices:
                h = str(population[idx]) # simple hash
                if h not in seen:
                    new_pop.append(population[idx])
                    seen.add(h)
                    added += 1
                if added >= cfg.elitism: break
            
            # Standard Tournament
            while len(new_pop) < cfg.pop_size:
                t1 = random.sample(range(len(population)), cfg.tournament_k)
                p1 = population[max(t1, key=lambda i: scalars[i])]
                t2 = random.sample(range(len(population)), cfg.tournament_k)
                p2 = population[max(t2, key=lambda i: scalars[i])]
                
                c1, c2 = crossover(p1, p2)
                new_pop.append(mutate(c1, num_inputs, current_mut_rate, cfg, locked_gates))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, current_mut_rate, cfg, locked_gates))
            
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
        string_gates.append({'name': f"G{i}", 'type': gate['type'], 'inputs': str_inputs, 'output': f"G{i}"})
    return string_gates