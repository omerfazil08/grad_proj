# evolution_scaled.py
# Phase 1: scalable GA with adaptive population and diversity injection
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from logic_gates import GATES
from grad_34_simp2 import simplify_single_output


# ---------------------------
# Config
# ---------------------------
@dataclass
class GAConfig:
    # Search size
    num_gates: int = 8
    pop_size_start: int = 150
    pop_size_min: int = 80

    # Evolution
    generations: int = 800
    elitism: int = 4
    parent_pool_topk: int = 10

    # Mutation (adaptive decay)
    base_mutation: float = 0.30  # decays to ~0 over generations

    # Diversity injection: every N gens, replace worst X% with random
    diversity_every: int = 50
    diversity_fraction: float = 0.10  # 10%

    # Stagnation-triggered population shrink
    stagnation_window: int = 120        # gens without improvement
    shrink_factor: float = 0.85         # new_pop = int(pop * factor), floored at pop_size_min

    # Progress print cadence
    log_every: int = 20

    # Reproducibility
    seed: int = 42


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
    # Base inputs: A, B, (optional) C
    base = [chr(65 + i) for i in range(num_inputs)]  # A,B,(C)...
    # We maintain compatibility with your 2-input flow by still having C/nC present (C will be 0)
    if num_inputs == 2:
        base = ['A', 'B', 'C']  # C injected as constant input path (the tuple passes c=0)
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


def evaluate_fitness(individual: List[Dict[str, Any]],
                     num_outputs: int,
                     inputs: List[Tuple[int, int, int]],
                     targets: List[List[int]]) -> int:
    score = 0
    for idx, (a, b, c) in enumerate(inputs):
        signals = evaluate_network(individual, a, b, c)
        for o in range(num_outputs):
            out_name = individual[-num_outputs + o]['name']
            if signals[out_name] == targets[o][idx]:
                score += 1
    return score


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

    for i, gate in enumerate(individual):
        if random.random() < mutation_rate:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)

    return mutant


def crossover(p1: List[Dict[str, Any]], p2: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(p1) < 3:  # safety for tiny genomes
        return p1[:], p2[:]
    point = random.randint(1, len(p1) - 2)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


# ---------------------------
# Evolution loop (Phase 1)
# ---------------------------
def evolve_scaled(num_inputs: int,
                  num_outputs: int,
                  inputs: List[Tuple[int, int, int]],
                  targets: List[List[int]],
                  cfg: GAConfig) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, List[int]]]:

    random.seed(cfg.seed)

    pop_size = cfg.pop_size_start
    population = [random_individual(num_inputs, cfg.num_gates) for _ in range(pop_size)]
    max_score = len(inputs) * num_outputs

    best = None
    best_score = -1
    best_gen = -1

    history = {'gen': [], 'best': [], 'pop': []}
    last_improve_gen = 0

    for gen in range(cfg.generations):
        # Sort by current fitness
        population.sort(key=lambda ind: -evaluate_fitness(ind, num_outputs, inputs, targets))
        cur_best = population[0]
        cur_best_score = evaluate_fitness(cur_best, num_outputs, inputs, targets)

        # Log
        if gen % cfg.log_every == 0 or cur_best_score == max_score:
            print(f"Gen {gen:4d} | Best = {cur_best_score}/{max_score} | Pop = {pop_size}")

        # Track history
        history['gen'].append(gen)
        history['best'].append(cur_best_score)
        history['pop'].append(pop_size)

        # Keep global best
        if cur_best_score > best_score:
            best, best_score, best_gen = cur_best, cur_best_score, gen
            last_improve_gen = gen

        # Success
        if cur_best_score == max_score:
            break

        # Diversity injection (periodic)
        if cfg.diversity_every > 0 and (gen > 0) and (gen % cfg.diversity_every == 0):
            inject_count = max(1, int(pop_size * cfg.diversity_fraction))
            for i in range(inject_count):
                population[-(i+1)] = random_individual(num_inputs, cfg.num_gates)

        # Stagnation-triggered shrink
        if (gen - last_improve_gen) >= cfg.stagnation_window and pop_size > cfg.pop_size_min:
            new_size = max(cfg.pop_size_min, int(pop_size * cfg.shrink_factor))
            # keep the top performers when shrinking
            population = population[:new_size]
            pop_size = new_size
            last_improve_gen = gen  # reset the stagnation window after shrink

        # Next generation (elitism + children)
        next_gen = population[:cfg.elitism]  # carry elite

        # Fill
        while len(next_gen) < pop_size:
            # Fast parent selection: choose from top-k (keeps your fast behavior)
            p1 = random.choice(population[:cfg.parent_pool_topk])
            p2 = random.choice(population[:cfg.parent_pool_topk])
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1, gen, cfg))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(c2, gen, cfg))

        population = next_gen

    return best, best_score, max_score, history


# ---------------------------
# Niceties
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
