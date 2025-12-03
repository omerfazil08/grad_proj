# evolution_colab_phase23.py
# Phase 2.3: Incremental Evolution with Statistical Gradient Analysis
# Based on evolution_colab_phase2.py

import random
import time
import json
import statistics  # <--- NEW: For Gradient Analysis
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from multiprocessing import Pool, cpu_count

# --- Configuration Class ---
class ColabConfig:
    def __init__(
        self,
        num_gates: int,
        pop_size: int,
        generations: int,
        elitism: int,
        tournament_k: int,
        base_mut: float,
        min_mut: float,
        p_choose_primitive: float,
        log_every: int,
        record_history: bool,
        seed: Optional[int],
        size_penalty_lambda: float,
        parallel: bool = True,
        processes: Optional[int] = None,
    ):
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
        self.processes = processes


# --- Logic Gate Definitions ---
def AND(*args): return int(all(args))
def OR(*args): return int(any(args))
def NOT(a): return int(not a)
def XOR2(a, b): return int(a != b)
def XNOR2(a, b): return int(a == b)
def NAND(*args): return int(not all(args))
def NOR(*args): return int(not any(args))
def EQ1(a, b): return int(a == b)
def MUX2(s, a, b): return a if s == 0 else b
def HALF_SUM(a, b): return XOR2(a, b)
def HALF_CARRY(a, b): return AND(a, b)
def FULL_SUM(a, b, cin): return XOR2(XOR2(a, b), cin)
def FULL_CARRY(a, b, cin): return OR(AND(a, b), AND(cin, XOR2(a, b)))


# --- Global Gate Set ---
GATES_SET = {
    "AND":        {"func": AND,        "arity": "variable", "min_arity": 2},
    "OR":         {"func": OR,         "arity": "variable", "min_arity": 2},
    "NOT":        {"func": NOT,        "arity": 1},
    "XOR2":       {"func": XOR2,       "arity": 2},
    "XNOR2":      {"func": XNOR2,      "arity": 2},
    "NAND":       {"func": NAND,       "arity": "variable", "min_arity": 2},
    "NOR":        {"func": NOR,        "arity": "variable", "min_arity": 2},
    "EQ1":        {"func": EQ1,        "arity": 2},
    "MUX2":       {"func": MUX2,       "arity": 3},
    "HALF_SUM":   {"func": HALF_SUM,   "arity": 2},
    "HALF_CARRY": {"func": HALF_CARRY, "arity": 2},
    "FULL_SUM":   {"func": FULL_SUM,   "arity": 3},
    "FULL_CARRY": {"func": FULL_CARRY, "arity": 3},
}


# --- Helper: Naming and Sources ---
def gate_name(idx: int) -> str:
    return f"G{idx}"

def allowed_sources(num_inputs: int, current_gate_idx: int) -> List[str]:
    sources = [f"A{i}" for i in range(num_inputs)]
    sources.extend([gate_name(i) for i in range(current_gate_idx)])
    return sources


# --- HSS Helpers ---
def hammersley_point(i: int, n: int, dims: int) -> List[float]:
    def radical_inverse(base: int, index: int) -> float:
        inv, denom = 0.0, 1.0
        while index > 0:
            index, rem = divmod(index, base)
            denom *= base
            inv += rem / denom
        return inv

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    pt = [i / max(1, n)]
    for d in range(dims - 1):
        b = primes[d % len(primes)]
        pt.append(radical_inverse(b, i))
    return pt

def hss_unit_cube(n_points: int, dims: int) -> List[List[float]]:
    return [hammersley_point(i, n_points, dims) for i in range(n_points)]

def _hss_take(vec: List[float], idx: int) -> Tuple[float, int]:
    v = vec[idx % len(vec)]
    return v, idx + 1


def random_gate_hss(num_inputs: int, gate_idx: int, cfg: ColabConfig,
                    vec: List[float], idx: int) -> Tuple[Dict[str, Any], int]:
    srcs = allowed_sources(num_inputs, gate_idx)
    possible_gate_types = []
    for gate_type, gate_info in GATES_SET.items():
        if gate_info["arity"] == "variable":
            if len(srcs) >= gate_info["min_arity"]:
                possible_gate_types.append(gate_type)
        else:
            if len(srcs) >= gate_info["arity"]:
                possible_gate_types.append(gate_type)

    if not possible_gate_types:
        if srcs:
            gate_type = "NOT"; k = 1
        else:
            gate_type = "AND"; k = 2
            srcs = [f"A{i % max(1, num_inputs)}" for i in range(k)]
    else:
        v, idx = _hss_take(vec, idx)
        gate_type = possible_gate_types[int(v * len(possible_gate_types)) % len(possible_gate_types)]
        gate_info = GATES_SET[gate_type]

        if gate_info["arity"] == "variable":
            min_ar = gate_info["min_arity"]
            max_ar = min(3, len(srcs))
            if max_ar < min_ar: k = min_ar
            else:
                v, idx = _hss_take(vec, idx)
                k = min_ar + int(v * (max_ar - min_ar + 1)) % (max_ar - min_ar + 1)
        else:
            k = int(gate_info["arity"])

    ins = []
    for _ in range(k):
        v, idx = _hss_take(vec, idx)
        ins.append(srcs[int(v * len(srcs)) % len(srcs)])

    gate = {
        "name": gate_name(gate_idx),
        "type": gate_type,
        "inputs": ins,
        "output": gate_name(gate_idx),
    }
    return gate, idx

def hss_individual(num_inputs: int, cfg: ColabConfig, vec: List[float]) -> List[Dict[str, Any]]:
    indiv = []
    idx = 0
    for gi in range(cfg.num_gates):
        g, idx = random_gate_hss(num_inputs, gi, cfg, vec, idx)
        indiv.append(g)
    return indiv

def init_population_hss(num_inputs: int, cfg: ColabConfig) -> List[List[Dict[str, Any]]]:
    dims = max(10, 5 * cfg.num_gates)
    hss_vectors = hss_unit_cube(cfg.pop_size, dims)
    return [hss_individual(num_inputs, cfg, hss_vectors[i]) for i in range(cfg.pop_size)]


# --- Evaluation ---
def evaluate_network(individual: List[Dict[str, Any]], inputs: Tuple[int, ...]) -> Dict[str, int]:
    values = {f"A{i}": val for i, val in enumerate(inputs)}

    for gate in individual:
        gate_type = gate["type"]
        gate_func = GATES_SET[gate_type]["func"]
        gate_inputs = gate["inputs"]
        out_name = gate["output"]

        try:
            resolved_inputs = [values[src] for src in gate_inputs]
        except KeyError:
            values[out_name] = 0
            continue
        values[out_name] = gate_func(*resolved_inputs)
    return values


# --- PHASE 2 FITNESS: MULTI-OBJECTIVE TRACKING ---
def fitness_breakdown(individual: List[Dict[str, Any]],
                      num_inputs: int,
                      num_outputs: int,
                      inputs_set: List[Tuple[int, ...]],
                      targets_set: List[List[int]]) -> List[int]:
    """
    Returns a list of scores, one for each output channel.
    E.g. [16, 12, 8] if Out0 is perfect, Out1 has 12 correct, Out2 has 8.
    """
    if len(individual) < num_outputs:
        return [-1] * num_outputs

    scores = [0] * num_outputs
    output_gate_names = [gate["output"] for gate in individual[-num_outputs:]]

    for row_idx, inp in enumerate(inputs_set):
        evaluated = evaluate_network(individual, inp)
        
        for out_idx in range(num_outputs):
            actual = evaluated.get(output_gate_names[out_idx], 0)
            target = targets_set[out_idx][row_idx]
            if actual == target:
                scores[out_idx] += 1
                
    return scores


# --- Parallel Fitness Helpers ---
_PE_num_inputs = None
_PE_num_outputs = None
_PE_inputs_set = None
_PE_targets_set = None
_PE_solved_mask = None  # New global for Phase 2
_PE_max_per_col = 0

def _pe_init(num_inputs, num_outputs, inputs_set, targets_set, solved_mask):
    global _PE_num_inputs, _PE_num_outputs, _PE_inputs_set, _PE_targets_set, _PE_solved_mask, _PE_max_per_col
    _PE_num_inputs = num_inputs
    _PE_num_outputs = num_outputs
    _PE_inputs_set = inputs_set
    _PE_targets_set = targets_set
    _PE_solved_mask = solved_mask
    _PE_max_per_col = len(inputs_set)

def _pe_fitness_wrapper(individual):
    # 1. Get breakdown
    scores = fitness_breakdown(individual, _PE_num_inputs, _PE_num_outputs, 
                               _PE_inputs_set, _PE_targets_set)
    
    # 2. Calculate scalar fitness with "Don't Break the Chain" logic
    scalar = 0
    for i, s in enumerate(scores):
        scalar += s
        # If this column is officially solved, but this individual broke it:
        if _PE_solved_mask[i] and s < _PE_max_per_col:
            scalar -= 5000 # Huge penalty: dead on arrival
            
    return scalar, scores

# --- Selection, Crossover, Mutation (Standard) ---
def select_parent_tournament(population, fitness_scores, cfg):
    n = len(population)
    k = max(1, min(cfg.tournament_k, n))
    indices = random.sample(range(n), k)
    scores = [fitness_scores[i] for i in indices]
    best_idx = indices[scores.index(max(scores))]
    return population[best_idx]

def crossover(parent1, parent2):
    n = min(len(parent1), len(parent2))
    if n <= 1: return parent1[:], parent2[:]
    cp = random.randint(1, n - 1)
    return parent1[:cp] + parent2[cp:], parent2[:cp] + parent1[cp:]

def mutate(individual, num_inputs, mut_rate, cfg):
    new_ind = []
    for gi, gate in enumerate(individual):
        if random.random() < mut_rate:
            srcs = allowed_sources(num_inputs, gi)
            possible_gate_types = []
            for gate_type, gate_info in GATES_SET.items():
                if gate_info["arity"] == "variable":
                    if len(srcs) >= gate_info["min_arity"]: possible_gate_types.append(gate_type)
                else:
                    if len(srcs) >= gate_info["arity"]: possible_gate_types.append(gate_type)

            if not possible_gate_types:
                new_ind.append(gate); continue

            gate_type = random.choice(possible_gate_types)
            gate_info = GATES_SET[gate_type]

            if gate_info["arity"] == "variable":
                max_possible = min(3, len(srcs))
                if max_possible < gate_info["min_arity"]: new_ind.append(gate); continue
                k = random.randint(gate_info["min_arity"], max_possible)
            else:
                k = int(gate_info["arity"])

            ins = random.sample(srcs, k)
            new_ind.append({"name": gate_name(gi), "type": gate_type, "inputs": ins, "output": gate_name(gi)})
        else:
            new_ind.append(gate)
    return new_ind


# --- PHASE 2 EVOLUTION LOOP: INCREMENTAL GOALS + STATS ---
def evolve_colab_phase2(
    num_inputs: int,
    num_outputs: int,
    inputs_set: List[Tuple[int, ...]],
    targets_set: List[List[int]],
    cfg: ColabConfig,
):
    if cfg.seed is not None: random.seed(cfg.seed)

    print(f"Initializing population (HSS)...")
    population = init_population_hss(num_inputs, cfg)
    max_score_per_col = len(inputs_set)
    total_max_score = max_score_per_col * num_outputs
    
    # Hall of Fame to store partial solutions
    hall_of_fame = {} # {output_index: best_genome}
    solved_mask = [False] * num_outputs 

    history = defaultdict(list)
    # Ensure keys exist
    for k in ['gen', 'best', 'mu', 'sigma']:
        history[k] = []

    # Pool Setup
    pool = None
    if cfg.parallel:
        procs = cfg.processes or cpu_count()
        pool = Pool(processes=procs, initializer=_pe_init, 
                    initargs=(num_inputs, num_outputs, inputs_set, targets_set, solved_mask))

    try:
        for gen in range(cfg.generations):
            t0 = time.time()
            
            # Evaluate
            if cfg.parallel and pool:
                chunk = max(1, len(population) // ((cfg.processes or cpu_count()) * 2) or 1)
                results = pool.map(_pe_fitness_wrapper, population, chunksize=chunk)
                # results is list of (scalar, [scores])
                fitness_scalars = [r[0] for r in results]
                breakdowns = [r[1] for r in results]
            else:
                # Serial fallback
                fitness_scalars = []
                breakdowns = []
                for ind in population:
                    sc = fitness_breakdown(ind, num_inputs, num_outputs, inputs_set, targets_set)
                    scalar = sum(sc)
                    for i, s in enumerate(sc):
                        if solved_mask[i] and s < max_score_per_col:
                            scalar -= 5000
                    fitness_scalars.append(scalar)
                    breakdowns.append(sc)

            # Find best
            best_val = max(fitness_scalars)
            best_idx = fitness_scalars.index(best_val)
            best_ind = population[best_idx]
            best_bkd = breakdowns[best_idx] 

            # --- START STATISTICAL ANALYSIS ---
            # Calculate Mu (Mean)
            mu = statistics.mean(fitness_scalars)
            
            # Calculate Sigma (Standard Deviation)
            if len(fitness_scalars) > 1:
                sigma = statistics.stdev(fitness_scalars)
            else:
                sigma = 0.0
                
            # Log to history
            history['gen'].append(gen)
            history['best'].append(best_val)
            history['mu'].append(mu)
            history['sigma'].append(sigma)
            # --- END STATISTICAL ANALYSIS ---

            # Check for NEWLY solved outputs
            new_solve = False
            for i, score in enumerate(best_bkd):
                if score == max_score_per_col and not solved_mask[i]:
                    print(f"🎉 Output #{i+1} SOLVED at Gen {gen}! Locking it in.")
                    solved_mask[i] = True
                    hall_of_fame[i] = best_ind
                    new_solve = True
            
            if new_solve and cfg.parallel:
                pool.close(); pool.join()
                pool = Pool(processes=procs, initializer=_pe_init, 
                            initargs=(num_inputs, num_outputs, inputs_set, targets_set, solved_mask))

            # Logging
            if gen % cfg.log_every == 0:
                status_str = " ".join([f"Out{i}:{s}/{max_score_per_col}" for i,s in enumerate(best_bkd)])
                # New logging with stats
                print(f"Gen {gen:4d} | Best={best_val:.1f} | Avg={mu:.1f} | Sig={sigma:.2f} | {status_str}")

            if all(solved_mask):
                print(f"✅ All outputs solved! (Gen {gen})")
                break

            # Selection
            new_population = []
            
            # Elitism
            elite_indices = sorted(range(len(population)), key=lambda k: fitness_scalars[k], reverse=True)[:cfg.elitism]
            for i in elite_indices: new_population.append(population[i])
            
            # Hall of Fame Injection
            for i in range(num_outputs):
                if i in hall_of_fame:
                    new_population.append(hall_of_fame[i])

            while len(new_population) < cfg.pop_size:
                p1 = select_parent_tournament(population, fitness_scalars, cfg)
                p2 = select_parent_tournament(population, fitness_scalars, cfg)
                c1, c2 = crossover(p1, p2)
                
                rate = cfg.base_mut * (1 - gen / cfg.generations) + cfg.min_mut
                
                new_population.append(mutate(c1, num_inputs, rate, cfg))
                if len(new_population) < cfg.pop_size:
                    new_population.append(mutate(c2, num_inputs, rate, cfg))

            population = new_population

    finally:
        if pool: pool.close(); pool.join()

    return best_ind, best_bkd, hall_of_fame, history # Return history!

def print_results_phase2(best_ind, best_bkd, hall_of_fame, inputs_set, targets_set):
    print("\n=== Final Result ===")
    print(f"Final Scores: {best_bkd}")
    print("Hall of Fame (Specialists):")
    for k, v in hall_of_fame.items():
        print(f"  Output {k+1} Solved By: Individual with {len(v)} gates")