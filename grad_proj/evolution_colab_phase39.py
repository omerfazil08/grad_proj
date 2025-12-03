# evolution_colab_phase38.py
# Phase 3.8: "Defragmentation Mode"
# Solves an output -> Simplifies Logic -> Rewrites Genome to Indexes 0..k -> Wipes Rest -> Continues.

import random
import time
import statistics
import math
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
    print("⚠️ Warning: simplifier_phase14 not found. Defragmentation disabled.")

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

# --- Logic Gates ---
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

def init_population(num_inputs, cfg):
    pop = []
    for _ in range(cfg.pop_size):
        indiv = [random_gate(i, num_inputs, cfg.p_choose_primitive) for i in range(cfg.num_gates)]
        pop.append(indiv)
    return pop

# --- Mutation ---
def mutate(ind, num_inputs, rate, cfg, locked_count):
    new_ind = []
    for i, gate in enumerate(ind):
        # If index < locked_count, it is a Defagmented/Solved gate. DO NOT TOUCH.
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

def fitness_bitwise(individual, packed_inputs, packed_targets, mask, num_inputs, solved_output_map):
    # solved_output_map: {target_idx: gate_index}
    # If an output is "Solved & Locked", we read it from the specific gate index.
    # If it is not solved, we read it from the standard "Last N" gates.
    
    signals = evaluate_bitwise(individual, packed_inputs, mask, num_inputs)
    
    num_gates = len(individual)
    num_targets = len(packed_targets)
    
    scores = []
    
    for i, target in enumerate(packed_targets):
        # Determine which signal corresponds to this output
        if i in solved_output_map:
            # It is locked at a specific position (e.g. G4 is Output 1)
            gate_idx = solved_output_map[i]
            # Signal index = num_inputs + gate_idx
            sig_idx = num_inputs + gate_idx
        else:
            # Standard "End of Chain" mapping
            # Output 0 is last, Output 1 is 2nd last... (Reverse order)
            # Or Standard Forward: Output 0 is -N, Output 1 is -(N-1)
            # Let's stick to: Outputs are the LAST N gates in order.
            # Output 0 -> ind[ -num_targets + 0 ]
            sig_idx = num_inputs + (num_gates - num_targets + i)
            
        if sig_idx < len(signals):
            out_val = signals[sig_idx]
        else:
            out_val = 0
            
        diff = out_val ^ target
        scores.append(mask.bit_count() - diff.bit_count())
        
    return scores

# --- Parallel ---
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

# --- DEFRAGMENTATION LOGIC ---

def synthesize_expr_to_gates(expr, num_inputs, gate_offset_start):
    generated_gates = []
    def get_sig(node):
        s = str(node)
        if s.startswith("A") and s[1:].isdigit(): return int(s[1:])
        return None
    def visit(node):
        sig = get_sig(node)
        if sig is not None: return sig
        if isinstance(node, Not):
            c = visit(node.args[0])
            new_idx = num_inputs + gate_offset_start + len(generated_gates)
            generated_gates.append({'name': f"G{new_idx-num_inputs}", 'type': 'NOT', 'inputs': [c]})
            return new_idx
        if isinstance(node, (And, Or, Xor)):
            op = "AND" if isinstance(node, And) else "OR" if isinstance(node, Or) else "XOR"
            args = [visit(a) for a in node.args]
            curr = args[0]
            for i in range(1, len(args)):
                new_idx = num_inputs + gate_offset_start + len(generated_gates)
                generated_gates.append({'name': f"G{new_idx-num_inputs}", 'type': op, 'inputs': [curr, args[i]]})
                curr = new_idx
            return curr
        return 0
    final_sig = visit(expr)
    return generated_gates, final_sig

def defragment_genome(best_ind, solved_indices, num_inputs, num_gates):
    """
    Takes the best individual.
    1. Identifies logic for newly solved output.
    2. Compresses it.
    3. Packs it at the start of the new genome.
    4. Returns (new_genome_template, new_locked_count, new_output_map)
    """
    if not HAS_SIMPLIFIER: return best_ind, 0, {}

    # 1. Convert to string for SymPy
    string_gates = []
    for i, g in enumerate(best_ind):
        s_ins = [f"A{x}" if x < num_inputs else f"G{x-num_inputs}" for x in g['inputs']]
        string_gates.append({'name': f"G{i}", 'type': g['type'], 'inputs': s_ins, 'output': f"G{i}"})
    
    input_names = [f"A{i}" for i in range(num_inputs)]
    
    # We rebuild the genome from scratch
    new_genome_structure = []
    new_output_map = {} # {target_idx: gate_idx}
    
    current_offset = 0
    
    # For every solved output, we extract and compact its logic
    for out_i in solved_indices:
        # Identify the gate currently driving this output
        # Standard mapping: Last N gates
        # CAUTION: If we already defragmented before, the mapping might be in the OLD output map?
        # For this iteration, let's assume we just solved 'out_i' and it sits at the standard position.
        
        # Actually, it's safer to ask the Simplifier to simplify the specific gate index
        # corresponding to output `out_i`.
        # Standard mapping: index = len(best_ind) - total_outputs + out_i
        # But wait, we need 'total_outputs' count. Let's assume we process `solved_indices` in order.
        
        # Since this is hard to map perfectly dynamically, let's use the "Trace Active" logic 
        # we built in Phase 3.4 but redirect the output to a new compressed block.
        
        target_gate_idx = len(best_ind) - len(solved_indices) + solved_indices.index(out_i) # Approximation
        # Correct approach: Passing the GATE NAME from the string list
        # Output indices are relative to the END of the list.
        # If we have 6 outputs, Output 0 is at -6, Output 5 is at -1.
        
        # This is complex. Let's try a simpler strategy for Phase 3.8:
        # ONLY compress the NEWLY solved output. 
        # Keep previously locked stuff as is? No, that defeats defragmentation.
        pass

    # --- IMPLEMENTATION OF PHASE 3.8 ---
    # We will simply TRACE the active gates of the newly solved output,
    # MOVE them to the front, re-index everything, and wipe the rest.
    # This does not require SymPy (safer) and achieves defragmentation.
    
    # 1. Identify Active Gates for all SOLVED outputs
    # (Union of all required gates)
    needed_gate_indices = set()
    output_gate_indices = {} # target_i -> old_gate_idx
    
    for out_i in solved_indices:
        # Find standard position
        # Assumes `solved_indices` passed here implies we check all 6 outputs?
        # Let's pass `total_outputs` to this function in future.
        # Workaround: fitness_bitwise logic: sig_idx = num_inputs + (num_gates - num_targets + i)
        # We need num_targets. 
        
        # Let's just return a placeholder since integrating this deeply requires changing the signature
        # of `evolve_bitwise`.
        pass

    return best_ind, 0, {} # Fallback


# --- Main Loop ---
def evolve_bitwise(num_inputs, num_outputs, inputs_list, targets_list, cfg):
    random.seed(cfg.seed)
    packed_inputs, mask = pack_truth_table(inputs_list, num_inputs)
    packed_targets = pack_targets(targets_list)
    max_score_per_col = mask.bit_count()
    
    print(f"Bitwise v3.8: Defragmentation Enabled.")
    
    population = init_population(num_inputs, cfg)
    solved_mask = [False] * num_outputs
    
    # Tracks which gate index holds the solution for which output
    # {0: 5, 1: 12} -> Output 0 is at G5, Output 1 is at G12
    solved_output_map = {} 
    
    # How many gates at the start are frozen
    locked_count = 0
    
    history = {'gen': [], 'best': [], 'mu': [], 'sigma': []}
    last_solve_gen = 0
    
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
            newly_solved_indices = []
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}!")
                    solved_mask[i] = True
                    new_solve = True
                    newly_solved_indices.append(i)

            # --- DEFRAGMENTATION TRIGGER ---
            if new_solve:
                best_ind = population[best_idx]
                
                # 1. Identify all gates needed for ALL currently solved outputs
                # This consolidates logic.
                keep_indices = set()
                
                # Re-map outputs. 
                # If an output was already solved/mapped, its gate is in solved_output_map.
                # If it is NEWLY solved, it is at the standard end position.
                
                temp_output_map = solved_output_map.copy()
                
                for out_i in range(num_outputs):
                    if solved_mask[out_i]:
                        # Where is the gate?
                        if out_i in solved_output_map:
                            gate_idx = solved_output_map[out_i]
                        else:
                            # Standard position: End of list
                            gate_idx = len(best_ind) - num_outputs + out_i
                        
                        # Trace dependencies
                        stack = [gate_idx]
                        keep_indices.add(gate_idx)
                        
                        while stack:
                            curr = stack.pop()
                            g = best_ind[curr]
                            for inp in g['inputs']:
                                if inp >= num_inputs: # Internal gate
                                    src_idx = inp - num_inputs
                                    if src_idx not in keep_indices:
                                        keep_indices.add(src_idx)
                                        stack.append(src_idx)
                                        
                        # We will map this output to its new position later
                        temp_output_map[out_i] = gate_idx # Placeholder
                
                # 2. Compact the Genome
                # Sort indices to preserve topological order
                sorted_keep = sorted(list(keep_indices))
                
                # Map old_index -> new_index (0, 1, 2...)
                remap = {old: new for new, old in enumerate(sorted_keep)}
                
                new_genome = []
                for old_idx in sorted_keep:
                    gate = best_ind[old_idx].copy()
                    # Remap inputs
                    new_inputs = []
                    for inp in gate['inputs']:
                        if inp >= num_inputs:
                            src = inp - num_inputs
                            # If the input isn't in kept list (shouldn't happen if traced well), map to 0
                            new_inputs.append(num_inputs + remap.get(src, 0))
                        else:
                            new_inputs.append(inp) # Primary input
                    
                    gate['inputs'] = new_inputs
                    gate['name'] = f"G{len(new_genome)}"
                    new_genome.append(gate)
                
                # 3. Update Output Map
                final_output_map = {}
                for out_i in range(num_outputs):
                    if solved_mask[out_i]:
                        # Where was it?
                        if out_i in solved_output_map:
                            old_loc = solved_output_map[out_i]
                        else:
                            old_loc = len(best_ind) - num_outputs + out_i
                        
                        final_output_map[out_i] = remap[old_loc]
                
                # 4. Fill rest with random junk
                locked_count = len(new_genome)
                print(f"   >> Defrag: Compressed {len(keep_indices)} essential gates into indices 0-{locked_count-1}.")
                
                while len(new_genome) < cfg.num_gates:
                    idx = len(new_genome)
                    new_genome.append(random_gate(idx, num_inputs, cfg.p_choose_primitive))
                
                # 5. Overwrite Population & Globals
                for i in range(len(population)):
                    population[i] = [g.copy() for g in new_genome] # Deep copy
                
                solved_output_map = final_output_map
                
                # Update pool
                if pool:
                    pool.close(); pool.join()
                    pool = Pool(processes=cpu_count(), initializer=_init_pool, 
                                initargs=(packed_inputs, packed_targets, mask, num_inputs, solved_output_map))
            
            # Logging
            if gen % cfg.log_every == 0:
                print(f"Gen {gen:4d} | Best={best_val} | Locked={locked_count} | {best_bkd}")
                
            if all(solved_mask):
                print(f"✅ All outputs solved at Gen {gen}!")
                return population[0], best_bkd, {}, history

            # Reproduction
            new_pop = []
            # Elitism (Just take best)
            new_pop.extend([population[best_idx]] * cfg.elitism)
            
            while len(new_pop) < cfg.pop_size:
                p1 = population[random.randint(0, len(population)-1)]
                p2 = population[random.randint(0, len(population)-1)]
                c1, c2 = crossover(p1, p2)
                # Mutate with lock protection
                new_pop.append(mutate(c1, num_inputs, cfg.base_mut, cfg, locked_count))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(mutate(c2, num_inputs, cfg.base_mut, cfg, locked_count))
            
            population = new_pop

    finally:
        if pool: pool.close(); pool.join()

    return population[0], best_bkd, {}, history

def convert_to_string_format(individual, num_inputs):
    string_gates = []
    for i, gate in enumerate(individual):
        str_inputs = []
        for inp_idx in gate['inputs']:
            if inp_idx < num_inputs: str_inputs.append(f"A{inp_idx}")
            else: str_inputs.append(f"G{inp_idx - num_inputs}")
        string_gates.append({'name': f"G{i}", 'type': gate['type'], 'inputs': str_inputs, 'output': f"G{i}"})
    return string_gates