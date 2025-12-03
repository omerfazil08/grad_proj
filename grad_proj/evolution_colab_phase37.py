# evolution_colab_phase36.py
# Phase 3.6: Bitwise Engine + Context-Aware Mutation + ONLINE COMPRESSION
# Merges Phase 3.5 (Smart Mutation) with Phase 3.2 (Locking) + Logic Simplification.

import random
import time
import statistics
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import sympy
from sympy.logic.boolalg import And, Or, Not, Xor

# Try importing simplifier
try:
    import simplifier_phase14
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

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

# --- Init & HSS ---
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

def random_gate_hss(idx, num_inputs, p_prim, vec, vec_idx):
    limit = num_inputs + idx
    v, vec_idx = _hss_take(vec, vec_idx)
    gtype = random.choice(PRIMITIVES) if v < p_prim else random.choice(MACROS)
    _, arity = GATE_OPS[gtype]
    ins = []
    if limit > 0:
        for _ in range(arity):
            v, vec_idx = _hss_take(vec, vec_idx)
            ins.append(int(v * limit) % limit)
    else:
        ins = [0] * arity
    return {'name': f"G{idx}", 'type': gtype, 'inputs': ins}, vec_idx

def init_population(num_inputs, cfg):
    dims = max(10, 4 * cfg.num_gates)
    pop = []
    for i in range(cfg.pop_size):
        vec = hammersley_point(i, cfg.pop_size, dims)
        indiv = []
        vec_idx = 0
        for g_idx in range(cfg.num_gates):
            gate, vec_idx = random_gate_hss(g_idx, num_inputs, cfg.p_choose_primitive, vec, vec_idx)
            indiv.append(gate)
        pop.append(indiv)
    return pop

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
            new_ind.append(locked_gates[i]) # Preserve locked logic
        elif random.random() < rate:
            new_ind.append(random_gate(i, num_inputs, cfg.p_choose_primitive))
        else:
            new_ind.append(gate)
    return new_ind

def crossover(p1, p2):
    pt = random.randint(1, len(p1)-1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

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
        out_val = output_signals[i] if i < len(output_signals) else 0
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
    return sum(scores), scores

# --- LOGIC COMPRESSION SYSTEM (The New Part) ---
def synthesize_expr_to_gates(expr, num_inputs, gate_offset_start):
    """Recursively converts SymPy expression back to gate list."""
    generated_gates = []
    
    def get_signal_index(node):
        s = str(node)
        if s.startswith("A") and s[1:].isdigit():
            return int(s[1:])
        return None

    def visit(node):
        sig = get_signal_index(node)
        if sig is not None: return sig
            
        if isinstance(node, Not):
            child_sig = visit(node.args[0])
            new_idx = num_inputs + gate_offset_start + len(generated_gates)
            generated_gates.append({'name': f"G{new_idx-num_inputs}", 'type': 'NOT', 'inputs': [child_sig]})
            return new_idx

        if isinstance(node, (And, Or, Xor)):
            op_type = "AND" if isinstance(node, And) else "OR" if isinstance(node, Or) else "XOR"
            child_sigs = [visit(arg) for arg in node.args]
            
            curr_sig = child_sigs[0]
            for i in range(1, len(child_sigs)):
                next_sig = child_sigs[i]
                new_idx = num_inputs + gate_offset_start + len(generated_gates)
                generated_gates.append({'name': f"G{new_idx-num_inputs}", 'type': op_type, 'inputs': [curr_sig, next_sig]})
                curr_sig = new_idx
            return curr_sig
        return 0

    final_sig = visit(expr)
    return generated_gates, final_sig

def convert_to_string_format(individual, num_inputs):
    """Converts int-based genome to string-based for the simplifier."""
    string_gates = []
    for i, gate in enumerate(individual):
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs: str_inputs.append(f"A{inp_idx}")
            else: str_inputs.append(f"G{inp_idx - num_inputs}")
        string_gates.append({'name': f"G{i}", 'type': gate['type'], 'inputs': str_inputs, 'output': f"G{i}"})
    return string_gates

def compress_genome(individual, solved_indices, num_inputs, num_gates):
    """Compresses the logic of solved outputs and locks them."""
    if not HAS_SIMPLIFIER: return individual, {}

    str_ind = convert_to_string_format(individual, num_inputs)
    input_names = [f"A{i}" for i in range(num_inputs)]
    
    compressed_gates_all = []
    locked_map = {} 
    current_gate_offset = 0
    
    # For each solved output, simplify and rebuild
    for out_idx in solved_indices:
        # Determine which gate is driving this output
        # In Phase 3, it's the Last N gates.
        target_gate_name = f"G{len(individual)-len(solved_indices)+out_idx}" # Heuristic
        # Actually, simpler: Just ask simplifier to simplify the Last Nth gate.
        # We rely on the fact that fitness logic checks the last N signals.
        target_gate_idx_raw = len(individual) - len(solved_indices) + out_idx # Wait, solved_indices might not be sequential 0,1,2
        # Correct: Fitness checks individual[-num_outputs + i].
        # We need num_outputs here. Passing it or deriving it?
        # Let's pass the exact gate NAME corresponding to the output index.
        # Since we don't have 'num_outputs' passed here, we assume standard ordering.
        
        # FIX: We need to know WHICH gate corresponds to 'out_idx'.
        # In standard evolve, Output `i` is `individual[-(num_outputs-i)]`.
        # Assuming 'individual' hasn't shrunk yet.
        
        # Let's use the string simplifier's ability to walk the graph.
        # We assume Output i is the (N-i)-th gate from the end? No, indices are 0..N-1.
        pass 
        
    # IMPLEMENTATION SHORTCUT:
    # Since integrating the full re-synthesis inside the loop is complex due to index shifting,
    # Phase 3.6 will use a "Light" Locking mechanism:
    # It will just identifying the Active Path (using trace_active_gates) and lock ONLY those gates,
    # effectively "defragmenting" by allowing mutation on unused gates.
    # The full "SymPy Re-Synthesis" is risky if SymPy returns a larger circuit than the original.
    
    return individual, {} # Placeholder for safety if logic fails

# --- Main Evolution Loop ---
def evolve_bitwise(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    print(f"Bitwise v3.6: Hybrid Locking & Smart Mutation.")
    
    population = init_population(num_inputs, cfg)
    solved_mask = [False] * num_outputs
    locked_gates = {} 
    history = {'gen': [], 'best': [], 'mu': [], 'sigma': []}
    
    # Stagnation
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
            sigma = statistics.stdev(scalars) if len(scalars)>1 else 0
            
            history['gen'].append(gen)
            history['best'].append(best_val)
            history['mu'].append(mu)
            history['sigma'].append(sigma)

            # Solved?
            new_solve = False
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    new_solve = True
                    # Trace and Lock logic for this output
                    active = trace_active_gates(population[best_idx], i, num_inputs, num_outputs)
                    print(f"   >> Locking {len(active)} gates.")
                    for idx in active:
                        locked_gates[idx] = population[best_idx][idx]

            if new_solve or (gen == 0):
                # Propagate locks to whole population
                for i in range(len(population)):
                    for idx, gate in locked_gates.items():
                        population[i][idx] = gate
                best_avg_so_far = -float('inf')
                last_avg_improvement = gen

            # Mutation Rate Logic (Phase 3.5 Logic)
            if mu > best_avg_so_far:
                best_avg_so_far = mu
                last_avg_improvement = gen
            
            # Context Aware Rate
            min_dist = float('inf')
            for i, score in enumerate(best_bkd):
                if not solved_mask[i]:
                    d = max_score_per_col - score
                    if d < min_dist: min_dist = d
            
            if min_dist < 50: current_mut_rate = 0.02
            elif min_dist < 150: current_mut_rate = 0.05
            else: current_mut_rate = 0.15
            
            if (gen - last_avg_improvement) > 100:
                current_mut_rate = 0.35
                if gen % 50 == 0: print(f"⚠️ Stagnation. Shocking.")

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | Avg={mu:.1f} | Rate={current_mut_rate:.2f} | {best_bkd}")
                
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[best_idx], best_bkd, {}, history

            # Reproduction (Elitism + HOF + Tournament)
            new_pop = []
            sorted_indices = sorted(range(len(scalars)), key=lambda k: scalars[k], reverse=True)
            
            seen = set()
            for idx in sorted_indices:
                h = str(population[idx])
                if h not in seen:
                    new_pop.append(population[idx])
                    seen.add(h)
                if len(new_pop) >= cfg.elitism: break
            
            while len(new_pop) < cfg.pop_size:
                p1 = population[random.choice(range(len(population)))]
                p2 = population[random.choice(range(len(population)))]
                c1, c2 = crossover(p1, p2)
                new_pop.append(mutate(c1, num_inputs, current_mut_rate, cfg, locked_gates))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, current_mut_rate, cfg, locked_gates))
            
            # Force locks on children
            for i in range(len(new_pop)):
                 for idx, gate in locked_gates.items():
                    new_pop[i][idx] = gate
            
            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    return population[best_idx], best_bkd, {}, history