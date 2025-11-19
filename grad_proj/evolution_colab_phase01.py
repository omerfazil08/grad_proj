%%writefile /content/grad_proj/evolution_colab_phase01.py
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional


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
        parallel: bool = False,
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

        # Present for compatibility with phase11 / mains,
        # but NOT used for actual parallelism in this file.
        self.parallel = parallel
        self.processes = processes


# --- Logic Gate Definitions ---
def AND(*args):
    return int(all(args))

def OR(*args):
    return int(any(args))

def NOT(a):
    return int(not a)

def XOR2(a, b):
    return int(a != b)

def XNOR2(a, b):
    return int(a == b)

def NAND(*args):
    return int(not all(args))

def NOR(*args):
    return int(not any(args))

def EQ1(a, b):
    return int(a == b)

def MUX2(s, a, b):
    return a if s == 0 else b

def HALF_SUM(a, b):
    return XOR2(a, b)

def HALF_CARRY(a, b):
    return AND(a, b)

def FULL_SUM(a, b, cin):
    return XOR2(XOR2(a, b), cin)

def FULL_CARRY(a, b, cin):
    return OR(AND(a, b), AND(cin, XOR2(a, b)))


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
    sources = [f"A{i}" for i in range(num_inputs)]             # primary inputs
    sources.extend([gate_name(i) for i in range(current_gate_idx)])  # previous gates
    return sources


# --- HSS (Hammersley Sequence Sampling) Helpers ---
def hammersley_point(i: int, n: int, dims: int) -> List[float]:
    """
    Hammersley point in [0,1]^dims using i/n and Van der Corput components.
    """
    def radical_inverse(base: int, index: int) -> float:
        inv, denom = 0.0, 1.0
        while index > 0:
            index, rem = divmod(index, base)
            denom *= base
            inv += rem / denom
        return inv

    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    pt = [i / max(1, n)]
    for d in range(dims - 1):
        b = primes[d % len(primes)]
        pt.append(radical_inverse(b, i))
    return pt

def hss_unit_cube(n_points: int, dims: int) -> List[List[float]]:
    return [hammersley_point(i, n_points, dims) for i in range(n_points)]

def _hss_take(vec: List[float], idx: int) -> Tuple[float, int]:
    """Take one quasi-random value from vec, cycling if needed."""
    v = vec[idx % len(vec)]
    return v, idx + 1


# --- HSS-based Individual Creation ---
def random_gate_hss(num_inputs: int, gate_idx: int, cfg: ColabConfig,
                    vec: List[float], idx: int) -> Tuple[Dict[str, Any], int]:
    """
    Create a gate using HSS vector 'vec' instead of Python's random.*.
    Returns (gate_dict, new_vec_index).
    """
    srcs = allowed_sources(num_inputs, gate_idx)

    # Filter gates usable with these sources
    possible_gate_types = []
    for gate_type, gate_info in GATES_SET.items():
        if gate_info["arity"] == "variable":
            if len(srcs) >= gate_info["min_arity"]:
                possible_gate_types.append(gate_type)
        else:
            if len(srcs) >= gate_info["arity"]:
                possible_gate_types.append(gate_type)

    if not possible_gate_types:
        # Fallback (very rare): synthesize a NOT if possible, else dummy AND
        if srcs:
            gate_type = "NOT"
            k = 1
        else:
            gate_type = "AND"
            k = 2
            srcs = [f"A{i % max(1, num_inputs)}" for i in range(k)]
    else:
        v, idx = _hss_take(vec, idx)
        gate_type = possible_gate_types[int(v * len(possible_gate_types)) % len(possible_gate_types)]
        gate_info = GATES_SET[gate_type]

        if gate_info["arity"] == "variable":
            # Choose between min_arity..3 (bounded by len(srcs))
            min_ar = gate_info["min_arity"]
            max_ar = min(3, len(srcs))
            if max_ar < min_ar:
                max_ar = min_ar
            v, idx = _hss_take(vec, idx)
            k = min_ar + int(v * (max_ar - min_ar + 1)) % (max_ar - min_ar + 1)
        else:
            k = int(gate_info["arity"])

    # Pick k inputs using HSS (allowing duplicates is fine for GA)
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
    """
    Build initial population using HSS over a moderately high-dimensional cube.
    """
    dims = max(6, 3 * cfg.num_gates)
    hss_vectors = hss_unit_cube(cfg.pop_size, dims)
    return [hss_individual(num_inputs, cfg, hss_vectors[i]) for i in range(cfg.pop_size)]


# --- Evaluation ---
def evaluate_network(individual: List[Dict[str, Any]], inputs: Tuple[int, ...]) -> Dict[str, int]:
    # Map from source name (A0, G0, ...) to value
    values = {f"A{i}": val for i, val in enumerate(inputs)}

    for gate in individual:
        gate_type = gate["type"]
        gate_func = GATES_SET[gate_type]["func"]
        gate_inputs = gate["inputs"]
        out_name = gate["output"]

        try:
            resolved_inputs = [values[src] for src in gate_inputs]
        except KeyError as e:
            # Robustness fallback
            # If some source is missing (shouldn't happen often), set output to 0
            values[out_name] = 0
            continue

        values[out_name] = gate_func(*resolved_inputs)

    return values


# --- Fitness Calculation ---
def fitness(individual: List[Dict[str, Any]],
            num_inputs: int,
            num_outputs: int,
            inputs_set: List[Tuple[int, ...]],
            targets_set: List[List[int]],
            cfg: ColabConfig) -> float:

    if len(individual) < num_outputs:
        return -1.0 * (cfg.generations + 1)

    score = 0
    output_gate_names = [gate["output"] for gate in individual[-num_outputs:]]

    for row_idx, inp in enumerate(inputs_set):
        evaluated = evaluate_network(individual, inp)
        circuit_outputs = [evaluated.get(name, 0) for name in output_gate_names]

        for out_idx in range(num_outputs):
            if out_idx < len(targets_set) and row_idx < len(targets_set[out_idx]):
                if circuit_outputs[out_idx] == targets_set[out_idx][row_idx]:
                    score += 1

    size_penalty = cfg.size_penalty_lambda * len(individual)
    return score - size_penalty


# --- Selection, Crossover, Mutation ---
def select_parent_tournament(population: List[List[Dict[str, Any]]],
                             fitness_scores: List[float],
                             cfg: ColabConfig) -> List[Dict[str, Any]]:
    n = len(population)
    if n == 0:
        raise ValueError("Empty population in tournament selection.")

    k = max(1, min(cfg.tournament_k, n))
    indices = random.sample(range(n), k)
    scores = [fitness_scores[i] for i in indices]
    best_idx = indices[scores.index(max(scores))]
    return population[best_idx]

def crossover(parent1: List[Dict[str, Any]],
              parent2: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = min(len(parent1), len(parent2))
    if n <= 1:
        return parent1[:], parent2[:]

    cp = random.randint(1, n - 1)
    child1 = parent1[:cp] + parent2[cp:]
    child2 = parent2[:cp] + parent1[cp:]
    return child1, child2

def mutate(individual: List[Dict[str, Any]],
           num_inputs: int,
           mut_rate: float,
           cfg: ColabConfig) -> List[Dict[str, Any]]:
    new_ind = []
    for gi, gate in enumerate(individual):
        if random.random() < mut_rate:
            # Use normal random gate (not HSS) for mutation; diversity is good here
            srcs = allowed_sources(num_inputs, gi)
            # Filter possible gate types
            possible_gate_types = []
            for gate_type, gate_info in GATES_SET.items():
                if gate_info["arity"] == "variable":
                    if len(srcs) >= gate_info["min_arity"]:
                        possible_gate_types.append(gate_type)
                else:
                    if len(srcs) >= gate_info["arity"]:
                        possible_gate_types.append(gate_type)

            if not possible_gate_types:
                new_ind.append(gate)
                continue

            gate_type = random.choice(possible_gate_types)
            gate_info = GATES_SET[gate_type]

            if gate_info["arity"] == "variable":
                k = random.randint(gate_info["min_arity"], min(3, len(srcs)))
            else:
                k = int(gate_info["arity"])

            ins = random.sample(srcs, k)
            new_ind.append({
                "name": gate_name(gi),
                "type": gate_type,
                "inputs": ins,
                "output": gate_name(gi),
            })
        else:
            new_ind.append(gate)
    return new_ind


# --- Main Evolutionary Loop (HSS + serial fitness) ---
def evolve_colab_phase0(
    num_inputs: int,
    num_outputs: int,
    inputs_set: List[Tuple[int, ...]],
    targets_set: List[List[int]],
    cfg: ColabConfig,
) -> Tuple[List[Dict[str, Any]], float, float, Dict[str, List[Any]]]:

    if cfg.seed is not None:
        random.seed(cfg.seed)

    # HSS-based initialization:
    population = init_population_hss(num_inputs, cfg)

    max_score = float(len(inputs_set) * num_outputs)
    history = defaultdict(list)

    for gen in range(cfg.generations):
        fitness_scores = [
            fitness(ind, num_inputs, num_outputs, inputs_set, targets_set, cfg)
            for ind in population
        ]

        best_score = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_score)]
        avg_score = sum(fitness_scores) / len(fitness_scores)

        if cfg.record_history:
            history["gen"].append(gen)
            history["best"].append(best_score)
            history["avg"].append(avg_score)

        if gen % cfg.log_every == 0 or best_score == max_score:
            print(f"Gen {gen:4d} | Best {best_score}/{max_score} | Avg {avg_score:.2f}")

        if best_score == max_score:
            print("Reached perfect fitness; stopping early.")
            break

        # --- Produce next generation ---
        new_population: List[List[Dict[str, Any]]] = []

        # Elitism
        elite_indices = sorted(
            range(len(population)),
            key=lambda k: fitness_scores[k],
            reverse=True
        )[:cfg.elitism]
        for i in elite_indices:
            new_population.append(population[i])

        # Offspring
        while len(new_population) < cfg.pop_size:
            parent1 = select_parent_tournament(population, fitness_scores, cfg)
            parent2 = select_parent_tournament(population, fitness_scores, cfg)

            child1, child2 = crossover(parent1, parent2)

            # Adaptive mutation
            mutation_rate = cfg.base_mut - (cfg.base_mut - cfg.min_mut) * (gen / cfg.generations)
            mutation_rate = max(cfg.min_mut, mutation_rate)

            new_population.append(mutate(child1, num_inputs, mutation_rate, cfg))
            if len(new_population) < cfg.pop_size:
                new_population.append(mutate(child2, num_inputs, mutation_rate, cfg))

        population = new_population

    # Final evaluation
    final_fitness_scores = [
        fitness(ind, num_inputs, num_outputs, inputs_set, targets_set, cfg)
        for ind in population
    ]
    final_best_score = max(final_fitness_scores)
    final_best_individual = population[final_fitness_scores.index(final_best_score)]

    return final_best_individual, final_best_score, max_score, history


# --- Utility for Printing Results ---
def print_results(best_individual: List[Dict[str, Any]], score: float, max_score: float,
                  num_outputs: int, inputs_set: List[Tuple[int, ...]], targets_set: List[List[int]]) -> None:
    print("\n=== Best Individual ===")
    print(f"Score: {score}/{max_score}")
    print("Gates:")
    for gate in best_individual:
        inputs_str = ",".join(gate["inputs"])
        print(f"  {gate['name']}: {gate['type']}({inputs_str})")

    print("\nTruth Table Check:")
    output_gate_names = [gate["output"] for gate in best_individual[-num_outputs:]]

    for i, input_tuple in enumerate(inputs_set):
        evaluated_values = evaluate_network(best_individual, input_tuple)
        circuit_outputs = [evaluated_values.get(name, 0) for name in output_gate_names]
        target_outputs = [targets_set[j][i] for j in range(num_outputs)]
        print(f"{input_tuple} \u2192 {circuit_outputs}   (target {target_outputs})")
