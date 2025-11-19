# evolution_parallel.py
# Phase 2: parallel fitness, optional caching, fast/accurate mode, timing
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count

from logic_gates import GATES
from grad_34_simp2 import simplify_single_output


# ---------------------------
# Config
# ---------------------------
@dataclass
class GAConfig:
    # Search size / genome
    num_gates: int = 8
    pop_size_start: int = 150
    pop_size_min: int = 80

    # Evolution
    generations: int = 800
    elitism: int = 4
    parent_pool_topk: int = 10

    # Mutation (adaptive decay)
    base_mutation: float = 0.30

    # Diversity injection
    diversity_every: int = 50
    diversity_fraction: float = 0.10

    # Stagnation-triggered population shrink
    stagnation_window: int = 120
    shrink_factor: float = 0.85

    # Progress print cadence
    log_every: int = 20

    # Reproducibility
    seed: int = 42

    # Phase 2 toggles
    parallel: bool = True              # enable multiprocessing for fitness
    processes: int | None = None       # None -> use cpu_count()
    cache_enabled: bool = True         # enable genome fitness cache
    fast_mode: bool = True             # keep fast simple operators (no extra overhead)


# ---------------------------
# Genome helpers
# ---------------------------
def random_gate(index: int, available_inputs: List[str]) -> Dict[str, Any]:
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }


def random_individual(num_inputs: int, num_gates: int) -> List[Dict[str, Any]]:
    # Base inputs: A, B, (optional) C for compatibility with (a,b,c) tuples
    base = [chr(65 + i) for i in range(num_inputs)]  # A,B,(C)...
    if num_inputs == 2:
        base = ['A', 'B', 'C']  # keep C path; caller passes c=0
    available = base + [f'n{x}' for x in base]

    indiv = []
    for i in range(num_gates):
        g = random_gate(i, available)
        available.append(g['name'])
        indiv.append(g)
    return indiv


# ---------------------------
# Simulation / Fitness
# ---------------------------
def evaluate_network(individual: List[Dict[str, Any]], a: int, b: int, c: int) -> Dict[str, int]:
    signals = {
        'A': a, 'B': b, 'C': c,
        'nA': 1 - a, 'nB': 1 - b, 'nC': 1 - c
    }
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals


def _fitness_core(individual: List[Dict[str, Any]],
                  num_outputs: int,
                  inputs: List[Tuple[int, int, int]],
                  targets: List[List[int]]) -> int:
    score = 0
    # hot path: local variables
    ind = individual
    outs = num_outputs
    for idx, (a, b, c) in enumerate(inputs):
        sig = evaluate_network(ind, a, b, c)
        # last gates are the outputs in order
        base = len(ind) - outs
        # manual unroll for 1–2 outputs common case
        if outs == 1:
            if sig[ind[base]['name']] == targets[0][idx]:
                score += 1
        elif outs == 2:
            if sig[ind[base]['name']] == targets[0][idx]:
                score += 1
            if sig[ind[base + 1]['name']] == targets[1][idx]:
                score += 1
        else:
            for o in range(outs):
                if sig[ind[base + o]['name']] == targets[o][idx]:
                    score += 1
    return score


# ---------------------------
# Parallel evaluation helpers
# ---------------------------
# Globals used inside worker processes
_PE_inputs: List[Tuple[int, int, int]] = []
_PE_targets: List[List[int]] = []
_PE_num_outputs: int = 1


def _pe_init(inputs, targets, num_outputs):
    global _PE_inputs, _PE_targets, _PE_num_outputs
    _PE_inputs = inputs
    _PE_targets = targets
    _PE_num_outputs = num_outputs


def _pe_fitness(individual: List[Dict[str, Any]]) -> int:
    return _fitness_core(individual, _PE_num_outputs, _PE_inputs, _PE_targets)


# ---------------------------
# Variation operators
# ---------------------------
def mutate(individual: List[Dict[str, Any]],
           gen: int,
           cfg: GAConfig) -> List[Dict[str, Any]]:
    # adaptive decay
    mutation_rate = cfg.base_mutation * (1 - gen / max(1, cfg.generations))
    mutant = []
    available = ['A', 'B', 'C', 'nA', 'nB', 'nC']

    # fast mode: keep original simple mutation (replace whole gate)
    for i, gate in enumerate(individual):
        if random.random() < mutation_rate:
            new_gate = random_gate(i, available.copy())
        else:
            # shallow copy is fine; gate is a flat dict
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)

    return mutant


def crossover(p1: List[Dict[str, Any]], p2: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(p1) < 3:  # safety
        return p1[:], p2[:]
    point = random.randint(1, len(p1) - 2)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


# ---------------------------
# Fitness dispatchers (serial / parallel / cached)
# ---------------------------
def compute_fitnesses_serial(population: List[List[Dict[str, Any]]],
                             num_outputs: int,
                             inputs: List[Tuple[int, int, int]],
                             targets: List[List[int]],
                             cache: Dict[str, int] | None) -> List[int]:
    fits = []
    for ind in population:
        if cache is not None:
            key = repr(ind)
            f = cache.get(key)
            if f is None:
                f = _fitness_core(ind, num_outputs, inputs, targets)
                cache[key] = f
        else:
            f = _fitness_core(ind, num_outputs, inputs, targets)
        fits.append(f)
    return fits


def compute_fitnesses_parallel(population: List[List[Dict[str, Any]]],
                               num_outputs: int,
                               inputs: List[Tuple[int, int, int]],
                               targets: List[List[int]],
                               processes: int | None,
                               cache: Dict[str, int] | None) -> List[int]:
    # If cache is on, split into: cached + to_compute
    if cache is not None:
        keys = [repr(ind) for ind in population]
        missing_mask = [k not in cache for k in keys]
        to_compute = [population[i] for i, miss in enumerate(missing_mask) if miss]
        if to_compute:
            with Pool(processes=processes, initializer=_pe_init, initargs=(inputs, targets, num_outputs)) as pool:
                computed = pool.map(_pe_fitness, to_compute, chunksize=max(1, len(to_compute)//(processes or cpu_count()) or 1))
            # fill cache
            idx = 0
            for i, miss in enumerate(missing_mask):
                if miss:
                    cache[keys[i]] = computed[idx]
                    idx += 1
        return [cache[k] for k in keys]

    # No cache → compute all in parallel
    with Pool(processes=processes, initializer=_pe_init, initargs=(inputs, targets, num_outputs)) as pool:
        return pool.map(_pe_fitness, population, chunksize=max(1, len(population)//(processes or cpu_count()) or 1))


# ---------------------------
# Evolution loop (Phase 2)
# ---------------------------
def evolve_parallel(num_inputs: int,
                    num_outputs: int,
                    inputs: List[Tuple[int, int, int]],
                    targets: List[List[int]],
                    cfg: GAConfig) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, List[float]]]:

    random.seed(cfg.seed)

    pop_size = cfg.pop_size_start
    population = [random_individual(num_inputs, cfg.num_gates) for _ in range(pop_size)]
    max_score = len(inputs) * num_outputs

    best = None
    best_score = -1
    best_gen = -1

    history: Dict[str, List[float]] = {
        'gen': [], 'best': [], 'pop': [],
        't_eval_ms': [], 't_gen_ms': [], 't_total_ms': []
    }
    last_improve_gen = 0

    # cache
    cache: Dict[str, int] | None = {} if cfg.cache_enabled else None

    for gen in range(cfg.generations):
        t0 = time.perf_counter()

        # Evaluate (parallel or serial)
        te0 = time.perf_counter()
        if cfg.parallel:
            fits = compute_fitnesses_parallel(population, num_outputs, inputs, targets,
                                              processes=cfg.processes, cache=cache)
        else:
            fits = compute_fitnesses_serial(population, num_outputs, inputs, targets, cache=cache)
        te1 = time.perf_counter()

        # Rank
        ranked = sorted(zip(population, fits), key=lambda x: -x[1])
        cur_best, cur_best_score = ranked[0]

        # Logging
        if gen % cfg.log_every == 0 or cur_best_score == max_score:
            print(f"Gen {gen:4d} | Best = {cur_best_score}/{max_score} | Pop = {pop_size}")

        # Track history
        history['gen'].append(gen)
        history['best'].append(float(cur_best_score))
        history['pop'].append(float(pop_size))
        history['t_eval_ms'].append((te1 - te0) * 1000.0)

        # Update global best
        if cur_best_score > best_score:
            best, best_score, best_gen = cur_best, cur_best_score, gen
            last_improve_gen = gen

        # Success
        if cur_best_score == max_score:
            history['t_gen_ms'].append(0.0)
            history['t_total_ms'].append((time.perf_counter() - t0) * 1000.0)
            break

        # Diversity injection (periodic)
        if cfg.diversity_every > 0 and (gen > 0) and (gen % cfg.diversity_every == 0):
            inject_count = max(1, int(pop_size * cfg.diversity_fraction))
            # replace worst slice
            for i in range(inject_count):
                population[-(i+1)] = random_individual(num_inputs, cfg.num_gates)

        # Stagnation-triggered shrink
        if (gen - last_improve_gen) >= cfg.stagnation_window and pop_size > cfg.pop_size_min:
            new_size = max(cfg.pop_size_min, int(pop_size * cfg.shrink_factor))
            population = [p for p, _ in ranked[:new_size]]
            pop_size = new_size
            last_improve_gen = gen
            # When we shrink, cache still valid (genomes kept by repr)

        # Next generation
        tg0 = time.perf_counter()
        elites = [p for p, _ in ranked[:cfg.elitism]]
        next_gen = elites[:]

        # Parent selection: fast top-k sampling (keeps your fast behavior)
        while len(next_gen) < pop_size:
            p1 = ranked[random.randrange(cfg.parent_pool_topk)][0]
            p2 = ranked[random.randrange(cfg.parent_pool_topk)][0]
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1, gen, cfg))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(c2, gen, cfg))

        population = next_gen
        tg1 = time.perf_counter()

        # timing
        history['t_gen_ms'].append((tg1 - tg0) * 1000.0)
        history['t_total_ms'].append((time.perf_counter() - t0) * 1000.0)

    return best, best_score, max_score, history


# ---------------------------
# Output helpers
# ---------------------------
def print_results(best: List[Dict[str, Any]],
                  score: int,
                  max_score: int,
                  num_outputs: int,
                  inputs: List[Tuple[int, int, int]],
                  targets: List[List[int]]) -> None:

    print("\n✅ Best Network Found:", score, "/", max_score)
    print("GATE LIST:")
    for g in best:
        print(f"{g['name']}: {g['gate']}({g['inputs'][0]}, {g['inputs'][1]})")

    print("\nTruth Table Check:")
    for idx, (a, b, c) in enumerate(inputs):
        out = evaluate_network(best, a, b, c)
        row = [out[best[-num_outputs + o]['name']] for o in range(num_outputs)]
        print(f"{(a,b,c)} → {row}   (target {[targets[o][idx] for o in range(num_outputs)]})")

    print("\n🧠 Simplified Output Logic:")
    for o in range(num_outputs):
        output_gate = best[-num_outputs + o]["name"]
        simplified = simplify_single_output(best, output_gate)
        print(f"Output {o+1} ({output_gate}):\n  {simplified}")
