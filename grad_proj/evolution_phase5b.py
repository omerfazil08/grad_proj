# evolution_phase5b.py
# Phase 5b – Adaptive Macro-Decay Genetic Evolution System
# by Ömer Fazıl Orhan & ChatGPT (GPT-5)
#
# Adds: gradual macro fade-out (adaptive probability), same GA core as Turbo

import random
import math
from dataclasses import dataclass
from grad_34_simp6 import simplify_single_output

# ------------------------------------------------------------
# Gate definitions
# ------------------------------------------------------------
PRIMITIVES = ["AND", "OR", "XOR2", "XNOR2", "NAND", "NOR"]
MACROS = [
    "HALF_SUM", "HALF_CARRY", "FULL_SUM", "FULL_CARRY",
    "MUX2", "EQ1", "GT1", "LT1"
]
ALL_GATES = PRIMITIVES + MACROS


def random_gate(allow_macros_prob: float):
    """Return random gate type with macro probability control."""
    if random.random() < allow_macros_prob:
        return random.choice(PRIMITIVES)
    return random.choice(MACROS)


def gate_eval(name, ins):
    """Simple symbolic evaluation (truth-level)."""
    a = ins[0] if len(ins) > 0 else 0
    b = ins[1] if len(ins) > 1 else 0
    c = ins[2] if len(ins) > 2 else 0
    if name == "AND": return a & b
    if name == "OR": return a | b
    if name == "XOR2": return a ^ b
    if name == "XNOR2": return int(not (a ^ b))
    if name == "NAND": return int(not (a & b))
    if name == "NOR": return int(not (a | b))
    # Macro shortcuts
    if name == "HALF_SUM": return a ^ b
    if name == "HALF_CARRY": return a & b
    if name == "FULL_SUM": return (a ^ b) ^ c
    if name == "FULL_CARRY": return (a & b) | (a & c) | (b & c)
    if name == "MUX2": return (a if c == 0 else b)
    if name == "EQ1": return int(a == b)
    if name == "GT1": return int(a > b)
    if name == "LT1": return int(a < b)
    return 0


# ------------------------------------------------------------
# Config and helpers
# ------------------------------------------------------------
@dataclass
class GAConfig:
    pop_size: int = 800
    generations: int = 300
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elite_count: int = 10
    gates_per_net: int = 16
    # Macro decay schedule
    p_choose_primitive: float = 0.3
    macro_decay_start: int = 0
    macro_decay_end: int = 100


def make_random_network(num_inputs, cfg: GAConfig, macro_prob=0.3):
    """Initial random network."""
    net = []
    for i in range(cfg.gates_per_net):
        g = {
            "name": f"g{i}",
            "gate": random_gate(macro_prob),
            "inputs": random.sample(["A", "B", "C", "D", "nA", "nB", "nC", "nD"], k=random.randint(2, 3))
        }
        net.append(g)
    return net


def mutate_gate(gate, macro_prob):
    """Mutate a gate randomly."""
    if random.random() < 0.3:
        gate["gate"] = random_gate(macro_prob)
    if random.random() < 0.5:
        gate["inputs"] = random.sample(["A", "B", "C", "D", "nA", "nB", "nC", "nD"], k=random.randint(2, 3))
    return gate


# ------------------------------------------------------------
# Evaluation logic
# ------------------------------------------------------------
def evaluate_network(network, inputs):
    signals = {**inputs}
    for gate in network:
        ins = [signals.get(i, 0) for i in gate["inputs"]]
        signals[gate["name"]] = gate_eval(gate["gate"], ins)
    return signals


def evaluate_fitness(individual, num_outputs, inputs, targets):
    score = 0
    for i, inp in enumerate(inputs):
        sig = evaluate_network(individual, {f"A": inp[0], "B": inp[1], "C": inp[2] if len(inp) > 2 else 0,
                                            "D": inp[3] if len(inp) > 3 else 0,
                                            "nA": int(not inp[0]), "nB": int(not inp[1]),
                                            "nC": int(not (inp[2] if len(inp) > 2 else 0)),
                                            "nD": int(not (inp[3] if len(inp) > 3 else 0))})
        for j in range(num_outputs):
            if sig.get(f"g{len(individual)-1-j}", 0) == targets[i][j]:
                score += 1
    return score


# ------------------------------------------------------------
# Main evolution
# ------------------------------------------------------------
def evolve_phase5b(num_inputs, num_outputs, inputs, targets, cfg: GAConfig):
    max_score = len(inputs) * num_outputs
    population = [make_random_network(num_inputs, cfg, macro_prob=cfg.p_choose_primitive)
                  for _ in range(cfg.pop_size)]

    best = None
    best_score = -1

    for gen in range(cfg.generations):
        # adaptive macro probability (fade-out)
        if gen < cfg.macro_decay_start:
            macro_prob = cfg.p_choose_primitive
        elif gen > cfg.macro_decay_end:
            macro_prob = 1.0
        else:
            t = (gen - cfg.macro_decay_start) / (cfg.macro_decay_end - cfg.macro_decay_start)
            macro_prob = cfg.p_choose_primitive + t * (1.0 - cfg.p_choose_primitive)

        scores = [evaluate_fitness(ind, num_outputs, inputs, targets) for ind in population]

        # track best
        best_idx = max(range(len(population)), key=lambda i: scores[i])
        if scores[best_idx] > best_score:
            best = population[best_idx]
            best_score = scores[best_idx]

        # progress
        if gen % 20 == 0 or best_score == max_score:
            print(f"Gen {gen:4d} | Best {best_score}/{max_score} | Pop {len(population)} | MacroProb {macro_prob:.2f}")
        if best_score == max_score:
            break

        # selection + reproduction
        new_pop = []
        elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:cfg.elite_count]
        for ei in elite_indices:
            new_pop.append(population[ei])

        while len(new_pop) < cfg.pop_size:
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2, cfg.crossover_rate)
            child = [mutate_gate(dict(g), macro_prob) for g in child] if random.random() < cfg.mutation_rate else child
            new_pop.append(child)
        population = new_pop

    return best, best_score, max_score


def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1 if random.random() < 0.5 else p2
    cut = random.randint(1, min(len(p1), len(p2)) - 1)
    return p1[:cut] + p2[cut:]
