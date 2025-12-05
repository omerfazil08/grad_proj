# evolution_colab_phase76.py
# Phase 7.6: Dynamic Buffer + Novelty Injection + Logic Compression
# Fixed: Proper state transitions (Accuracy -> Novelty -> Secondary Accuracy -> Backtrack)

import random
import time
import statistics
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import copy
import sympy
from sympy.logic.boolalg import And, Or, Not, Xor
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

try:
    import simplifier_phase76
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False
    print("⚠️ Warning: simplifier_phase76 not found. Logic compression disabled.")

# --- Configuration ---
class BitwiseConfig:
    def __init__(self, gate_growth_buffer, pop_size, generations, elitism, tournament_k, 
                 base_mut, min_mut, p_choose_primitive, log_every, 
                 record_history, seed, size_penalty_lambda, parallel=True, num_islands=5):
        
        self.gate_growth_buffer = gate_growth_buffer
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
        
        # Phase 7.6 Timings
        self.novelty_threshold = 800        # Start Novelty
        self.novelty_duration = 400         # Run Novelty for this long
        self.backtrack_threshold = 1500     # If still failing, Backtrack
        # Timeline: 0-800 (Acc), 800-1200 (Nov), 1200-1500 (Sec Acc), 1500 (Reset)
        
        self.avg_patience_factor = 0.005 

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

# --- HSS Helpers ---
def radical_inverse(base, index):
    inv, denom = 0.0, 1.0
    while index > 0:
        index, rem = divmod(index, base)
        denom *= base
        inv += rem / denom
    return inv

def hammersley_point(i, n, dims):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    pt = [i / max(1, n)]
    for d in range(dims - 1):
        b = primes[d % len(primes)]
        pt.append(radical_inverse(b, i))
    return pt

def _hss_take(vec, idx):
    return vec[idx % len(vec)], idx + 1

# --- HSS Init ---
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

def hss_individual(num_inputs, target_size, p_prim, vec, locked_prefix=None):
    indiv = []
    vec_idx = 0
    start_idx = 0
    if locked_prefix:
        indiv = [g.copy() for g in locked_prefix]
        start_idx = len(locked_prefix)
    for i in range(start_idx, target_size):
        gate, vec_idx = random_gate_hss(i, num_inputs, p_prim, vec, vec_idx)
        indiv.append(gate)
    return indiv

def init_population_hss(num_inputs, cfg, size, current_genome_size, offset_start=0, total_pop=1000, locked_prefix=None):
    needed = current_genome_size - (len(locked_prefix) if locked_prefix else 0)
    dims = max(10, 4 * needed)
    hss_vectors = [hammersley_point(offset_start + i, total_pop, dims) for i in range(size)]
    return [hss_individual(num_inputs, current_genome_size, cfg.p_choose_primitive, v, locked_prefix) for v in hss_vectors]

# --- Mutation ---
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
    
    if current_target_idx in solved_output_map:
        sig_idx = num_inputs + solved_output_map[current_target_idx]
    else:
        sig_idx = num_inputs + (num_gates - num_targets + current_target_idx)
    
    out_val = signals[sig_idx] if sig_idx < len(signals) else 0
    
    if mode == "NOVELTY":
        return out_val, out_val # Signature
        
    diff = out_val ^ packed_targets[current_target_idx]
    current_score = max_score_per_col - diff.bit_count()
    
    penalty = 0
    for i in range(current_target_idx):
        if i in solved_output_map:
            s_idx = num_inputs + solved_output_map[i]
            val = signals[s_idx] if s_idx < len(signals) else 0
            if (val ^ packed_targets[i]) != 0: penalty += 50000

    return current_score - penalty, current_score

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

# --- COMPRESSION ---
def synthesize_expr_to_gates(expr, num_inputs, gate_offset_start):
    generated_gates = []
    def get_signal_index(node):
        s = str(node)
        if s.startswith("A") and s[1:].isdigit(): return int(s[1:])
        return None

    def visit(node):
        sig = get_signal_index(node)
        if sig is not None: return sig
        op = None
        if isinstance(node, Not):
            child = visit(node.args[0])
            new_idx = num_inputs + gate_offset_start + len(generated_gates)
            generated_gates.append({'name': f"G{new_idx-num_inputs}", 'type': 'NOT', 'inputs': [child]})
            return new_idx
        if isinstance(node, And): op = "AND"
        elif isinstance(node, Or): op = "OR"
        elif isinstance(node, Xor): op = "XOR"
        if op:
            children = [visit(arg) for arg in node.args]
            curr = children[0]
            for i in range(1, len(children)):
                nxt = children[i]
                new_idx = num_inputs + gate_offset_start + len(generated_gates)
                generated_gates.append({'name': f"G{new_idx-num_inputs}", 'type': op, 'inputs': [curr, nxt]})
                curr = new_idx
            return curr
        return 0
    final_sig = visit(expr)
    return generated_gates, final_sig

def convert_for_simplifier(individual, num_inputs):
    string_gates = []
    for i, gate in enumerate(individual):
        gtype = gate['type']
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs: str_inputs.append(f"A{inp_idx}")
            else: str_inputs.append(f"G{inp_idx - num_inputs}")
        string_gates.append({'name': f"G{i}", 'type': gtype, 'inputs': str_inputs, 'output': f"G{i}"})
    return string_gates

def trace_active_gates(individual, solved_indices, num_inputs, num_outputs):
    keep_indices = set()
    for out_i in solved_indices:
        gate_idx = len(individual) - num_outputs + out_i
        stack = [gate_idx]
        keep_indices.add(gate_idx)
        while stack:
            curr = stack.pop()
            if curr >= len(individual): continue
            g = individual[curr]
            for inp in g['inputs']:
                if inp >= num_inputs:
                    src_idx = inp - num_inputs
                    if src_idx not in keep_indices:
                        keep_indices.add(src_idx)
                        stack.append(src_idx)
    return sorted(list(keep_indices))

def run_simplifier_with_timeout(str_ind, input_names, target_gate_names, timeout_sec=5):
    with Pool(processes=1) as pool:
        res = pool.apply_async(simplifier_phase76.simplify_genome, (str_ind, input_names, target_gate_names))
        try:
            return res.get(timeout=timeout_sec)
        except Exception:
            pool.terminate(); raise 

def compress_and_lock_genome(best_ind, solved_indices, num_inputs, num_outputs):
    if not HAS_SIMPLIFIER:
        return defragment_genome_raw(best_ind, solved_indices, {}, num_inputs, num_outputs)
    try:
        active_indices = trace_active_gates(best_ind, solved_indices, num_inputs, num_outputs)
        active_genome = []
        remap = {old: new for new, old in enumerate(active_indices)}
        for old_idx in active_indices:
            gate = best_ind[old_idx].copy()
            new_inputs = [num_inputs + remap.get(inp - num_inputs, 0) if inp >= num_inputs else inp for inp in gate['inputs']]
            gate['inputs'] = new_inputs
            gate['name'] = f"G{len(active_genome)}"
            active_genome.append(gate)
        
        raw_count = len(active_genome)
        str_ind = convert_for_simplifier(active_genome, num_inputs)
        input_names = [f"A{i}" for i in range(num_inputs)]
        target_gate_names = []
        for out_i in solved_indices:
            orig_idx = len(best_ind) - num_outputs + out_i
            if orig_idx in remap: target_gate_names.append(f"G{remap[orig_idx]}")
            else: target_gate_names.append("G0")

        try:
            simplified_map = run_simplifier_with_timeout(str_ind, input_names, target_gate_names, timeout_sec=6)
        except Exception:
            print(f"   ⏳ Simplifier Error/Timeout. Using Raw Active Trace.")
            return defragment_genome_raw(best_ind, solved_indices, {}, num_inputs, num_outputs)
        
        compressed_gates_all = []
        new_output_map = {} 
        current_gate_offset = 0
        for i, out_idx in enumerate(solved_indices):
            key_match = None
            for k in simplified_map.keys():
                if f"Output {out_idx+1}" in k: key_match = k; break
            if not key_match: continue
            expr_str = simplified_map[key_match]
            transformations = (standard_transformations + (convert_xor,))
            expr_sym = parse_expr(expr_str, transformations=transformations)
            new_gates, final_sig = synthesize_expr_to_gates(expr_sym, num_inputs, current_gate_offset)
            for g in new_gates:
                g['name'] = f"G{current_gate_offset}"
                compressed_gates_all.append(g)
                current_gate_offset += 1
            new_output_map[out_idx] = current_gate_offset - 1

        sim_count = len(compressed_gates_all)
        print("-" * 60)
        print(f"   📊 SIMPLIFICATION REPORT: Raw={raw_count} vs SymPy={sim_count}")
        if sim_count <= raw_count:
            print(f"   ✅ Using Simplified.")
            print("-" * 60)
            return compressed_gates_all, sim_count, new_output_map
        else:
            print(f"   ❌ SymPy bigger. Using Raw.")
            print("-" * 60)
            return defragment_genome_raw(best_ind, solved_indices, {}, num_inputs, num_outputs)
    except Exception as e:
        print(f"   ⚠️ Compression failed ({e}). Using Raw.")
        return defragment_genome_raw(best_ind, solved_indices, {}, num_inputs, num_outputs)

def defragment_genome_raw(best_ind, solved_indices, solved_output_map, num_inputs, num_outputs):
    keep_indices = trace_active_gates(best_ind, solved_indices, num_inputs, num_outputs)
    remap = {old: new for new, old in enumerate(keep_indices)}
    new_genome = []
    for old_idx in keep_indices:
        gate = best_ind[old_idx].copy()
        gate['inputs'] = [num_inputs + remap.get(inp - num_inputs, 0) if inp >= num_inputs else inp for inp in gate['inputs']]
        gate['name'] = f"G{len(new_genome)}"
        new_genome.append(gate)
    final_output_map = {}
    for out_i in solved_indices:
        old = len(best_ind) - num_outputs + out_i
        final_output_map[out_i] = remap.get(old, 0)
    return new_genome, len(new_genome), final_output_map

# --- Main Loop ---
def evolve_phase76(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    num_islands = cfg.num_islands
    island_pop_size = cfg.pop_size // num_islands
    print(f"🏝️ Phase 7.6: Novelty-Seeded Accuracy + Rolling Buffer.")
    
    current_genome_size = cfg.gate_growth_buffer
    print(f"   >> Initial Genome Size: {current_genome_size} gates")
    
    islands = []
    for k in range(num_islands):
        offset = k * island_pop_size
        islands.append(init_population_hss(num_inputs, cfg, island_pop_size, current_genome_size, offset_start=offset, total_pop=cfg.pop_size))
    
    solved_mask = [False] * num_outputs
    solved_output_map = {}
    locked_count = 0
    current_target = 0
    
    checkpoints = {} 
    stagnation_counter = 0
    best_score_history = -1
    best_avg_history = -1
    evolution_mode = "ACCURACY"
    
    novelty_archive = []
    
    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))

    try:
        for gen in range(cfg.generations):
            flat_pop = [ind for island in islands for ind in island]
            results = pool.map(_eval_wrapper, flat_pop)
            
            offset = 0
            global_best_raw = -1
            global_best_ind = None
            global_mu = 0
            
            new_islands = []
            
            for i in range(num_islands):
                res_slice = results[offset : offset + island_pop_size]
                island_pop = islands[i]
                offset += island_pop_size
                
                if evolution_mode == "ACCURACY":
                    scalars = [r[0] for r in res_slice]
                    raws = [r[1] for r in res_slice]
                    global_mu += sum(raws)
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
                    
                    # Harvest Novelty
                    unique_indices = sorted(range(len(scalars)), key=lambda k: scalars[k], reverse=True)[:5]
                    for idx in unique_indices:
                        novelty_archive.append(copy.deepcopy(island_pop[idx]))
                    if len(novelty_archive) > 1000: novelty_archive = novelty_archive[-1000:]

                if evolution_mode == "NOVELTY":
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
                    local_best = max(scalars)
                    local_idx = scalars.index(local_best)
                    if raws[local_idx] > global_best_raw:
                        global_best_raw = raws[local_idx]
                        global_best_ind = island_pop[local_idx]

                new_pop = []
                sorted_pop = [x for _, x in sorted(zip(scalars, island_pop), key=lambda p: p[0], reverse=True)]
                new_pop.extend([g[:] for g in sorted_pop[:cfg.elitism]])
                while len(new_pop) < island_pop_size:
                    p1 = random.choice(island_pop); p2 = random.choice(island_pop)
                    c1, c2 = crossover(p1, p2)
                    new_pop.append(mutate(c1, num_inputs, cfg.base_mut, cfg, locked_count))
                    if len(new_pop) < island_pop_size: new_pop.append(mutate(c2, num_inputs, cfg.base_mut, cfg, locked_count))
                new_islands.append(new_pop)
            
            islands = new_islands
            global_mu /= (cfg.pop_size)

            # Success
            if global_best_raw == max_score_per_col:
                print(f"🎉 Output #{current_target+1} SOLVED at Gen {gen}!")
                solved_mask[current_target] = True
                indices = [k for k, x in enumerate(solved_mask) if x]
                
                print("   >> Compressing Logic...")
                l_genome, l_count, l_map = compress_and_lock_genome(global_best_ind, indices, num_inputs, num_outputs)
                checkpoints[current_target] = (l_genome, l_count, l_map)
                current_target += 1
                solved_output_map = l_map
                locked_count = l_count
                
                current_genome_size = locked_count + cfg.gate_growth_buffer
                print(f"   >> New Genome Size: {current_genome_size} (Locked: {locked_count}, Buffer: {cfg.gate_growth_buffer})")
                
                stagnation_counter = 0; best_score_history = -1; evolution_mode = "ACCURACY"
                novelty_archive = [] 
                
                if current_target >= num_outputs:
                    print("✅ ALL OUTPUTS SOLVED!")
                    return global_best_ind
                
                pool.close(); pool.join()
                pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                            initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
                
                print("   >> 🌊 HSS Global Reset...")
                islands = []
                for k in range(num_islands):
                    off = k * island_pop_size
                    islands.append(init_population_hss(num_inputs, cfg, island_pop_size, current_genome_size, offset_start=off, total_pop=cfg.pop_size, locked_prefix=l_genome))
                continue

            # Improvements
            if global_best_raw > best_score_history:
                best_score_history = global_best_raw
                stagnation_counter = 0
                best_avg_history = global_mu
                if evolution_mode == "NOVELTY":
                    print(f"   >> Novelty found better accuracy ({global_best_raw})! Switching back to ACCURACY.")
                    
                    # INJECTION
                    if len(novelty_archive) > 0:
                        print(f"   >> Injecting {len(novelty_archive)} unique circuits...")
                        chunk = max(1, len(novelty_archive) // num_islands)
                        for k in range(num_islands):
                            start = k * chunk
                            end = min((k + 1) * chunk, len(novelty_archive))
                            if start < len(novelty_archive):
                                payload = novelty_archive[start:end]
                                islands[k][-(len(payload)):] = copy.deepcopy(payload)
                    
                    evolution_mode = "ACCURACY"
                    novelty_archive = []
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
            else:
                stagnation_counter += 1
                if global_mu > best_avg_history: best_avg_history = global_mu

            # State Logic
            # 1. Enter Novelty
            if stagnation_counter > cfg.novelty_threshold and evolution_mode == "ACCURACY":
                if global_mu >= best_avg_history * (1.0 + cfg.avg_patience_factor):
                     stagnation_counter = int(cfg.novelty_threshold * 0.8)
                     best_avg_history = global_mu
                else:
                    print(f"⚠️ Stagnation ({stagnation_counter}). Switching to NOVELTY.")
                    evolution_mode = "NOVELTY"
                    novelty_archive = []
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "NOVELTY"))
            
            # 2. End Novelty -> Secondary Accuracy
            elif stagnation_counter > (cfg.novelty_threshold + cfg.novelty_duration) and evolution_mode == "NOVELTY":
                print(f"   >> Novelty time up. Switching to Secondary ACCURACY with Injection.")
                
                # INJECTION
                if len(novelty_archive) > 0:
                    print(f"   >> Injecting {len(novelty_archive)} unique circuits...")
                    chunk = max(1, len(novelty_archive) // num_islands)
                    for k in range(num_islands):
                        start = k * chunk
                        end = min((k + 1) * chunk, len(novelty_archive))
                        if start < len(novelty_archive):
                            payload = novelty_archive[start:end]
                            islands[k][-(len(payload)):] = copy.deepcopy(payload)
                
                evolution_mode = "ACCURACY"
                novelty_archive = []
                pool.close(); pool.join()
                pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                            initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
            
            # 3. Backtrack
            elif stagnation_counter > cfg.backtrack_threshold and current_target > 0:
                print(f"⚠️ HARD STAGNATION. BACKTRACKING.")
                revert_idx = max(0, current_target - 1)
                if revert_idx == 0: l_genome, l_count, l_map = [], 0, {}
                else: l_genome, l_count, l_map = checkpoints[revert_idx-1]
                
                current_target = revert_idx
                solved_output_map = l_map
                locked_count = l_count
                solved_mask[current_target] = False
                
                current_genome_size = locked_count + cfg.gate_growth_buffer
                print(f"   >> Reverted Size: {current_genome_size}")
                
                islands = []
                for k in range(num_islands):
                    off = k * island_pop_size
                    islands.append(init_population_hss(num_inputs, cfg, island_pop_size, current_genome_size, offset_start=off, total_pop=cfg.pop_size, locked_prefix=l_genome))
                        
                stagnation_counter = 0; best_score_history = -1; evolution_mode = "ACCURACY"
                novelty_archive = []
                pool.close(); pool.join()
                pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                            initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map, current_target, "ACCURACY"))
                continue

            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Target #{current_target+1} | Score: {global_best_raw}/{max_score_per_col} | Avg: {global_mu:.1f} | Mode: {evolution_mode}")

    finally:
        if pool: pool.close(); pool.join()

    return global_best_ind