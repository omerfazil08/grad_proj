# evolution_phase3.py
# Phase 3: Smarter Genetic Mechanics (pure Python, CPU-friendly)
import random
import math
from dataclasses import dataclass , field
from typing import List, Tuple, Dict, Any, Literal, Optional

from logic_gates import GATES
from grad_34_simp2 import simplify_single_output


# ============================================================
# Config objects
# ============================================================


@dataclass
class EvolutionStrategy:
    init: Literal["random", "hss"] = "hss"
    selection: Literal["topk", "tournament", "roulette"] = "tournament"
    crossover: Literal["one_point", "two_point", "uniform"] = "two_point"
    p_replace_gate: float = 0.40
    p_gate_type: float = 0.20
    p_rewire_one_input: float = 0.25
    p_swap_inputs: float = 0.15


@dataclass
class GAConfig:
    num_gates: int = 8
    pop_size_start: int = 500
    generations: int = 1000
    elitism: int = 4
    parent_pool_topk: int = 10
    tournament_k: int = 3
    base_mutation: float = 0.30
    min_mutation: float = 0.05
    diversity_every: int = 80
    diversity_fraction: float = 0.10
    log_every: int = 20
    seed: int = 42
    diversity_pairs: int = 64
    # ✅ Correctly inside the dataclass now:
    strategy: EvolutionStrategy = field(default_factory=EvolutionStrategy)


# ============================================================
# Utilities: genome encoding & diversity
# ============================================================
def encode_genome(individual: List[Dict[str, Any]]) -> str:
    # Compact, deterministic string for Hamming-like distance & caching
    # Example chunk: g0|XOR|A,nB;g1|AND|g0,B;...
    parts = []
    for g in individual:
        parts.append(f"{g['name']}|{g['gate']}|{g['inputs'][0]},{g['inputs'][1]}")
    return ";".join(parts)

def genome_distance(a: str, b: str) -> int:
    # Hamming distance over equal-length strings; pad if needed
    la, lb = len(a), len(b)
    if la < lb:
        a += "#" * (lb - la)
    elif lb < la:
        b += "#" * (la - lb)
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def average_population_diversity(pop: List[List[Dict[str, Any]]], pairs: int) -> float:
    if len(pop) < 2:
        return 0.0
    enc = [encode_genome(ind) for ind in pop]
    n = len(pop)
    pairs = min(pairs, n * (n - 1) // 2)
    if pairs <= 0:
        return 0.0
    total = 0
    for _ in range(pairs):
        i = random.randrange(n)
        j = random.randrange(n)
        while j == i:
            j = random.randrange(n)
        total += genome_distance(enc[i], enc[j])
    return total / pairs


# ============================================================
# Hammersley Sequence Sampling (HSS) for quasi-random init
# ============================================================
def radical_inverse(base: int, index: int) -> float:
    # Classic Van der Corput in given base
    inv = 0.0
    denom = 1.0
    while index > 0:
        index, rem = divmod(index, base)
        denom *= base
        inv += rem / denom
    return inv

def hammersley_point(i: int, n: int, dims: int) -> List[float]:
    # dims >= 2 recommended. First dim = i/n, others use first primes
    # Small prime list for dims up to ~10
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    point = [i / n]
    for d in range(dims - 1):
        base = primes[d] if d < len(primes) else 2
        point.append(radical_inverse(base, i))
    return point

def hss_unit_cube(n_points: int, dims: int) -> List[List[float]]:
    return [hammersley_point(i, max(1, n_points), dims) for i in range(n_points)]


# ============================================================
# Genome helpers (safe wiring only; no reordering)
# ============================================================
def _base_inputs(num_inputs: int) -> List[str]:
    # Allow up to 8 inputs: A..H (extend easily)
    letters = [chr(ord('A') + i) for i in range(num_inputs)]
    # Keep compatibility with earlier (2-input tasks pass c=0)
    return letters

def random_gate(index: int, available_inputs: List[str]) -> Dict[str, Any]:
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

def random_individual(num_inputs: int, num_gates: int) -> List[Dict[str, Any]]:
    base = _base_inputs(num_inputs)
    available = base + [f'n{x}' for x in base]
    indiv = []
    for i in range(num_gates):
        g = random_gate(i, available)
        available.append(g['name'])
        indiv.append(g)
    return indiv

def hss_individual(num_inputs: int, num_gates: int, hss_vec: List[float]) -> List[Dict[str, Any]]:
    # Map HSS vector deterministically to (gate_type, input choices) per gate.
    base = _base_inputs(num_inputs)
    indiv = []
    available = base + [f'n{x}' for x in base]

    # We’ll consume values from hss_vec cyclically
    idx = 0
    def take():
        nonlocal idx
        v = hss_vec[idx % len(hss_vec)]
        idx += 1
        return v

    gate_names = list(GATES.keys())
    for gi in range(num_gates):
        gate_type = gate_names[int(take() * len(gate_names)) % len(gate_names)]

        a_idx = int(take() * len(available)) % len(available)
        b_idx = int(take() * len(available)) % len(available)
        # ensure two (possibly equal allowed? usually yes) — keep it as-is
        g = {
            'name': f'g{gi}',
            'gate': gate_type,
            'inputs': [available[a_idx], available[b_idx]]
        }
        indiv.append(g)
        available.append(g['name'])
    return indiv


# ============================================================
# Simulation & Fitness
# ============================================================
def evaluate_network(individual: List[Dict[str, Any]], inputs_dict: Dict[str, int]) -> Dict[str, int]:
    # Build signals with complement lines
    signals = {k: inputs_dict[k] for k in inputs_dict}
    for k, v in list(inputs_dict.items()):
        signals[f"n{k}"] = 1 - v

    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

def evaluate_fitness(individual: List[Dict[str, Any]],
                     num_outputs: int,
                     inputs: List[Tuple[int, ...]],
                     targets: List[List[int]]) -> int:
    score = 0
    # Precompute output gate indexes (last gates)
    base_idx = len(individual) - num_outputs
    names = [individual[base_idx + o]['name'] for o in range(num_outputs)]

    for row_idx, row in enumerate(inputs):
        inputs_dict = {chr(ord('A') + i): row[i] for i in range(len(row))}
        sig = evaluate_network(individual, inputs_dict)
        for o in range(num_outputs):
            if sig[names[o]] == targets[o][row_idx]:
                score += 1
    return score


# ============================================================
# Selection operators
# ============================================================
def select_parents(population: List[List[Dict[str, Any]]],
                   fitnesses: List[int],
                   cfg: GAConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sel = cfg.strategy.selection
    if sel == "topk":
        # Fast: choose from top-k
        ranked = sorted(zip(population, fitnesses), key=lambda x: -x[1])
        idx1 = random.randrange(min(cfg.parent_pool_topk, len(ranked)))
        idx2 = random.randrange(min(cfg.parent_pool_topk, len(ranked)))
        return ranked[idx1][0], ranked[idx2][0]

    elif sel == "tournament":
        k = max(2, cfg.tournament_k)
        def tour_pick():
            cand_idx = random.sample(range(len(population)), min(k, len(population)))
            best_i = max(cand_idx, key=lambda i: fitnesses[i])
            return population[best_i]
        return tour_pick(), tour_pick()

    elif sel == "roulette":
        total = sum(max(0, f) for f in fitnesses) + 1e-9
        probs = [(max(0, f) / total) for f in fitnesses]
        def pick():
            r = random.random()
            acc = 0.0
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    return population[i]
            return population[-1]
        return pick(), pick()

    # fallback
    return random.choice(population), random.choice(population)


# ============================================================
# Crossover operators
# ============================================================
def crossover(p1: List[Dict[str, Any]],
              p2: List[Dict[str, Any]],
              mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(p1)
    if n < 3:
        return p1[:], p2[:]

    if mode == "one_point":
        cut = random.randint(1, n - 2)
        return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

    if mode == "two_point":
        a = random.randint(1, n - 2)
        b = random.randint(a + 1, n - 1)
        c1 = p1[:a] + p2[a:b] + p1[b:]
        c2 = p2[:a] + p1[a:b] + p2[b:]
        return c1, c2

    if mode == "uniform":
        c1, c2 = [], []
        for i in range(n):
            if random.random() < 0.5:
                c1.append(p1[i].copy())
                c2.append(p2[i].copy())
            else:
                c1.append(p2[i].copy())
                c2.append(p1[i].copy())
        return c1, c2

    # default
    cut = random.randint(1, n - 2)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]


# ============================================================
# Mutation operators (safe)
# ============================================================
def _available_after_index(num_inputs: int, index: int) -> List[str]:
    # Inputs available when creating gate at position index:
    base = _base_inputs(num_inputs)
    av = base + [f'n{x}' for x in base]
    # previous gate outputs:
    for gi in range(index):
        av.append(f"g{gi}")
    return av

def mutate_gate(g: Dict[str, Any],
                gate_index: int,
                num_inputs: int,
                cfg: GAConfig) -> Dict[str, Any]:
    # Mixture of safe mutations
    r = random.random()
    new_g = g.copy()

    # Gate type change
    if r < cfg.strategy.p_gate_type:
        new_g["gate"] = random.choice(list(GATES.keys()))
        return new_g

    # Swap inputs
    r -= cfg.strategy.p_gate_type
    if r < cfg.strategy.p_swap_inputs:
        a, b = new_g["inputs"]
        new_g["inputs"] = [b, a]
        return new_g

    # Rewire exactly one input
    r -= cfg.strategy.p_swap_inputs
    if r < cfg.strategy.p_rewire_one_input:
        available = _available_after_index(num_inputs, gate_index)
        which = random.randrange(2)
        new_g["inputs"][which] = random.choice(available)
        return new_g

    # Fallback to full replacement (safe)
    available = _available_after_index(num_inputs, gate_index)
    new_g = {
        'name': g['name'],
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available, 2)
    }
    return new_g

def mutate(individual: List[Dict[str, Any]],
           gen: int,
           cfg: GAConfig,
           num_inputs: int) -> List[Dict[str, Any]]:
    # Adaptive mutation schedule
    t = gen / max(1, cfg.generations)
    rate = max(cfg.min_mutation, cfg.base_mutation * (1 - t))

    mutant: List[Dict[str, Any]] = []
    for i, g in enumerate(individual):
        if random.random() < rate:
            choice = random.random()
            # Bias toward "replace gate" some portion of the time
            if choice < cfg.strategy.p_replace_gate:
                available = _available_after_index(num_inputs, i)
                new_gate = {
                    'name': g['name'],
                    'gate': random.choice(list(GATES.keys())),
                    'inputs': random.sample(available, 2)
                }
            else:
                new_gate = mutate_gate(g, i, num_inputs, cfg)
        else:
            new_gate = g.copy()
        mutant.append(new_gate)
    return mutant


# ============================================================
# Evolution loop (Phase 3)
# ============================================================
def evolve_phase3(num_inputs: int,
                  num_outputs: int,
                  inputs: List[Tuple[int, ...]],
                  targets: List[List[int]],
                  cfg: GAConfig):
    random.seed(cfg.seed)

    # --- Initialize population (random or HSS)
    pop_size = cfg.pop_size_start
    population: List[List[Dict[str, Any]]] = []
    if cfg.strategy.init == "random":
        population = [random_individual(num_inputs, cfg.num_gates) for _ in range(pop_size)]
    else:
        # HSS init: generate quasi-random unit-cube points and map to genomes
        dims = max(6, 3 * cfg.num_gates)  # plenty of dimensions to feed mapping
        hss = hss_unit_cube(pop_size, dims)
        for vec in hss:
            population.append(hss_individual(num_inputs, cfg.num_gates, vec))

    max_score = len(inputs) * num_outputs

    history = {
        'gen': [],
        'best': [],
        'diversity': [],
        'pop': [],
    }

    best, best_score, best_gen = None, -1, -1

    for gen in range(cfg.generations):
        # Fitness
        fitnesses = [evaluate_fitness(ind, num_outputs, inputs, targets) for ind in population]

        # Rank
        ranked = sorted(zip(population, fitnesses), key=lambda x: -x[1])
        cur_best, cur_best_score = ranked[0]

        # Diversity (sampled)
        div = average_population_diversity(population, cfg.diversity_pairs)

        if gen % cfg.log_every == 0 or cur_best_score == max_score:
            print(f"Gen {gen:4d} | Best {cur_best_score}/{max_score} | Pop {len(population)} | Div {div:.1f}")

        # Track
        history['gen'].append(gen)
        history['best'].append(cur_best_score)
        history['diversity'].append(div)
        history['pop'].append(len(population))

        if cur_best_score > best_score:
            best, best_score, best_gen = cur_best, cur_best_score, gen

        if cur_best_score == max_score:
            break

        # Diversity injection (optional but helpful for larger tasks)
        if cfg.diversity_every > 0 and gen > 0 and (gen % cfg.diversity_every == 0):
            inject = max(1, int(len(population) * cfg.diversity_fraction))
            for i in range(inject):
                # use same init strategy for injected
                if cfg.strategy.init == "random":
                    population[-(i+1)] = random_individual(num_inputs, cfg.num_gates)
                else:
                    dims = max(6, 3 * cfg.num_gates)
                    vec = hammersley_point(gen * inject + i + 1, max(1, len(population)), dims)
                    population[-(i+1)] = hss_individual(num_inputs, cfg.num_gates, vec)

        # Next generation
        elites = [p for p, _ in ranked[:cfg.elitism]]
        next_gen = elites[:]

        # Fill with children
        sel_mode = cfg.strategy.selection
        cx_mode = cfg.strategy.crossover
        while len(next_gen) < len(population):
            p1, p2 = select_parents(population, fitnesses, cfg)
            c1, c2 = crossover(p1, p2, cx_mode)
            next_gen.append(mutate(c1, gen, cfg, num_inputs))
            if len(next_gen) < len(population):
                next_gen.append(mutate(c2, gen, cfg, num_inputs))

        population = next_gen

    return best, best_score, max_score, history


# ============================================================
# Output helpers
# ============================================================
def print_results(best: List[Dict[str, Any]],
                  score: int,
                  max_score: int,
                  num_outputs: int,
                  inputs: List[Tuple[int, ...]],
                  targets: List[List[int]]) -> None:

    print("\n✅ Best Network Found:", score, "/", max_score)
    print("GATE LIST:")
    for g in best:
        print(f"{g['name']}: {g['gate']}({g['inputs'][0]}, {g['inputs'][1]})")

    print("\nTruth Table Check:")
    base_idx = len(best) - num_outputs
    out_names = [best[base_idx + o]['name'] for o in range(num_outputs)]
    for idx, row in enumerate(inputs):
        inputs_dict = {chr(ord('A') + i): row[i] for i in range(len(row))}
        out = evaluate_network(best, inputs_dict)
        row_vals = [out[name] for name in out_names]
        print(f"{tuple(row)} → {row_vals}   (target {[targets[o][idx] for o in range(num_outputs)]})")

    print("\n🧠 Simplified Output Logic:")
    for o in range(num_outputs):
        output_gate = best[-num_outputs + o]["name"]
        simplified = simplify_single_output(best, output_gate)
        print(f"Output {o+1} ({output_gate}):\n  {simplified}")
