# evolution_phase5a.py (final stable build)
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Literal

# =====================================================
# === Base Gates ======================================
# =====================================================
def AND(*xs): return int(all(xs))
def OR(*xs): return int(any(xs))
def XOR2(a, b): return a ^ b
def XNOR2(a, b): return int(1 - (a ^ b))
def NAND(*xs): return int(not all(xs))
def NOR(*xs): return int(not any(xs))

BASE_GATES = {
    "AND": {"fn": AND, "arity": "2or3"},
    "OR": {"fn": OR, "arity": "2or3"},
    "NAND": {"fn": NAND, "arity": "2or3"},
    "NOR": {"fn": NOR, "arity": "2or3"},
    "XOR": {"fn": XOR2, "arity": 2},
    "XNOR": {"fn": XNOR2, "arity": 2},
}

# =====================================================
# === Macro Gates =====================================
# =====================================================
def HALF_SUM(a, b): return a ^ b
def HALF_CARRY(a, b): return a & b
def FULL_SUM(a, b, c): return a ^ b ^ c
def FULL_CARRY(a, b, c): return int((a & b) | (a & c) | (b & c))
def MUX2(sel, a, b): return int(((1 - sel) & a) | (sel & b))
def EQ1(a, b): return int(1 - (a ^ b))
def GT1(a, b): return int(a & (1 - b))
def LT1(a, b): return int((1 - a) & b)

MACRO_GATES = {
    "HALF_SUM": {"fn": HALF_SUM, "arity": 2},
    "HALF_CARRY": {"fn": HALF_CARRY, "arity": 2},
    "FULL_SUM": {"fn": FULL_SUM, "arity": 3},
    "FULL_CARRY": {"fn": FULL_CARRY, "arity": 3},
    "MUX2": {"fn": MUX2, "arity": 3},
    "EQ1": {"fn": EQ1, "arity": 2},
    "GT1": {"fn": GT1, "arity": 2},
    "LT1": {"fn": LT1, "arity": 2},
}

ALL_GATE_NAMES = list(BASE_GATES.keys()) + list(MACRO_GATES.keys())

# =====================================================
# === Safe Evaluation with Arity Fix ==================
# =====================================================
def gate_arity(name: str) -> int:
    if name in BASE_GATES:
        a = BASE_GATES[name]["arity"]
        return 3 if a == "2or3" and random.random() < 0.5 else (2 if a == "2or3" else int(a))
    return MACRO_GATES[name]["arity"]

def gate_eval(name: str, inputs: List[int]) -> int:
    """Safely evaluate any gate; auto-trim/pad inputs."""
    if name in BASE_GATES:
        fn = BASE_GATES[name]["fn"]
        ar = BASE_GATES[name]["arity"]
        if ar == "2or3":
            if len(inputs) < 2: inputs = inputs * 2
            elif len(inputs) > 3: inputs = inputs[:3]
            return int(fn(*inputs))
        # strictly enforce 2-input for XOR/XNOR
        if len(inputs) > 2: inputs = inputs[:2]
        if len(inputs) < 2: inputs = inputs * 2
        return int(fn(*inputs))

    # macros
    fn = MACRO_GATES[name]["fn"]
    need = MACRO_GATES[name]["arity"]
    if len(inputs) < need:
        inputs = inputs + [inputs[-1]] * (need - len(inputs))
    elif len(inputs) > need:
        inputs = inputs[:need]
    return int(fn(*inputs))

# =====================================================
# === HSS Initialization ==============================
# =====================================================
def radical_inverse(base, index):
    inv, denom = 0.0, 1.0
    while index > 0:
        index, rem = divmod(index, base)
        denom *= base
        inv += rem / denom
    return inv

def hammersley_point(i, n, dims):
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    pt = [i / n]
    for d in range(dims - 1):
        base = primes[d % len(primes)]
        pt.append(radical_inverse(base, i))
    return pt

def hss_unit_cube(n, dims):
    return [hammersley_point(i, n, dims) for i in range(n)]

# =====================================================
# === GA Configuration =================================
# =====================================================
@dataclass
class EvolutionStrategy:
    init: Literal["random", "hss"] = "hss"
    selection: Literal["tournament", "roulette", "topk"] = "tournament"
    crossover: Literal["one_point", "two_point", "uniform"] = "two_point"

@dataclass
class GAConfig:
    num_gates: int = 16
    pop_size: int = 600
    generations: int = 1200
    elitism: int = 6
    tournament_k: int = 3
    parent_pool_topk: int = 12
    base_mutation: float = 0.30
    min_mutation: float = 0.05
    log_every: int = 20
    seed: int = 42
    strategy: EvolutionStrategy = field(default_factory=EvolutionStrategy)

# =====================================================
# === Helpers =========================================
# =====================================================
def _base_inputs(num_inputs):
    return [chr(ord('A') + i) for i in range(num_inputs)]

def _available_after_index(num_inputs, gate_index):
    base = _base_inputs(num_inputs)
    av = base + [f"n{x}" for x in base]
    for gi in range(gate_index):
        av.append(f"g{gi}")
    return av

def random_gate(index, available):
    name = random.choice(ALL_GATE_NAMES)
    ar = gate_arity(name)
    if not available: available = ["A", "B"]
    chosen = random.sample(available, k=min(ar, len(available)))
    while len(chosen) < ar:
        chosen.append(random.choice(available))
    return {"name": f"g{index}", "gate": name, "inputs": chosen}

def random_individual(num_inputs, num_gates):
    base = _base_inputs(num_inputs)
    available = base + [f"n{x}" for x in base]
    indiv = []
    for i in range(num_gates):
        g = random_gate(i, available)
        available.append(g["name"])
        indiv.append(g)
    return indiv

def hss_individual(num_inputs, num_gates, vec):
    base = _base_inputs(num_inputs)
    available = base + [f"n{x}" for x in base]
    names = ALL_GATE_NAMES
    idx = 0
    def take():
        nonlocal idx
        v = vec[idx % len(vec)]
        idx += 1
        return v
    indiv = []
    for i in range(num_gates):
        gname = names[int(take() * len(names)) % len(names)]
        ar = gate_arity(gname)
        ins = [available[int(take() * len(available)) % len(available)] for _ in range(ar)]
        g = {"name": f"g{i}", "gate": gname, "inputs": ins}
        available.append(g["name"])
        indiv.append(g)
    return indiv

# =====================================================
# === Simulation / Fitness ============================
# =====================================================
def evaluate_network(individual, inputs_dict):
    signals = dict(inputs_dict)
    for k, v in list(inputs_dict.items()):
        if not k.startswith("n"):
            signals[f"n{k}"] = 1 - v
    for gate in individual:
        in_vals = [signals[i] for i in gate["inputs"]]
        signals[gate["name"]] = gate_eval(gate["gate"], in_vals)
    return signals

def evaluate_fitness(individual, num_outputs, inputs, targets):
    score = 0
    base_idx = len(individual) - num_outputs
    out_names = [individual[base_idx + o]["name"] for o in range(num_outputs)]
    nin = len(inputs[0])
    for r, row in enumerate(inputs):
        inp = {chr(ord('A') + i): row[i] for i in range(nin)}
        sig = evaluate_network(individual, inp)
        for o in range(num_outputs):
            if sig[out_names[o]] == targets[o][r]:
                score += 1
    return score

# =====================================================
# === GA Core =========================================
# =====================================================
def select_parents(pop, fits, cfg):
    if cfg.strategy.selection == "tournament":
        k = max(2, cfg.tournament_k)
        def tour():
            cand = random.sample(range(len(pop)), min(k, len(pop)))
            best_i = max(cand, key=lambda i: fits[i])
            return pop[best_i]
        return tour(), tour()
    if cfg.strategy.selection == "topk":
        ranked = sorted(range(len(pop)), key=lambda i: -fits[i])
        pick = lambda: pop[random.choice(ranked[:cfg.parent_pool_topk])]
        return pick(), pick()
    # roulette fallback
    total = sum(max(0, f) for f in fits) + 1e-9
    probs = [max(0, f) / total for f in fits]
    def pick():
        r, s = random.random(), 0.0
        for i, p in enumerate(probs):
            s += p
            if r <= s: return pop[i]
        return pop[-1]
    return pick(), pick()

def crossover(p1, p2, mode):
    n = len(p1)
    if n < 3: return p1[:], p2[:]
    if mode == "one_point":
        cut = random.randint(1, n - 2)
        return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]
    if mode == "two_point":
        a, b = sorted(random.sample(range(1, n - 1), 2))
        return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]
    # uniform
    c1, c2 = [], []
    for i in range(n):
        if random.random() < 0.5:
            c1.append(p1[i].copy()); c2.append(p2[i].copy())
        else:
            c1.append(p2[i].copy()); c2.append(p1[i].copy())
    return c1, c2

def mutate_gate(g, gate_index, num_inputs):
    new_g = g.copy()
    r = random.random()
    if r < 0.25:
        new_g["gate"] = random.choice(ALL_GATE_NAMES)
    elif r < 0.5 and len(new_g["inputs"]) >= 2:
        i, j = random.sample(range(len(new_g["inputs"])), 2)
        new_g["inputs"][i], new_g["inputs"][j] = new_g["inputs"][j], new_g["inputs"][i]
    elif r < 0.75:
        av = _available_after_index(num_inputs, gate_index)
        if av:
            which = random.randrange(len(new_g["inputs"]))
            new_g["inputs"][which] = random.choice(av)
    desired = gate_arity(new_g["gate"])
    av = _available_after_index(num_inputs, gate_index)
    while len(new_g["inputs"]) < desired and av:
        new_g["inputs"].append(random.choice(av))
    if len(new_g["inputs"]) > desired:
        new_g["inputs"] = new_g["inputs"][:desired]
    return new_g

def mutate(ind, gen, cfg, num_inputs):
    rate = max(cfg.min_mutation, cfg.base_mutation * (1 - gen / max(1, cfg.generations)))
    out = []
    for i, g in enumerate(ind):
        if random.random() < rate:
            if random.random() < 0.5:
                av = _available_after_index(num_inputs, i)
                new_g = random_gate(i, av)
            else:
                new_g = mutate_gate(g, i, num_inputs)
        else:
            new_g = g.copy()
        out.append(new_g)
    return out

# =====================================================
# === Evolution Loop ==================================
# =====================================================
def evolve_phase5a(num_inputs, num_outputs, inputs, targets, cfg):
    random.seed(cfg.seed)
    if cfg.strategy.init == "random":
        pop = [random_individual(num_inputs, cfg.num_gates) for _ in range(cfg.pop_size)]
    else:
        hss = hss_unit_cube(cfg.pop_size, max(6, 3 * cfg.num_gates))
        pop = [hss_individual(num_inputs, cfg.num_gates, v) for v in hss]

    max_score = len(inputs) * num_outputs
    best, best_score = None, -1

    for gen in range(cfg.generations):
        fits = [evaluate_fitness(ind, num_outputs, inputs, targets) for ind in pop]
        best_idx = max(range(len(pop)), key=lambda i: fits[i])
        cur_best, cur_score = pop[best_idx], fits[best_idx]
        if gen % cfg.log_every == 0 or cur_score == max_score:
            avg = sum(fits) / len(fits)
            print(f"Gen {gen:4d} | Best {cur_score}/{max_score} | Avg {avg:.2f}")
        if cur_score > best_score:
            best, best_score = cur_best, cur_score
        if cur_score == max_score: break

        ranked = sorted(range(len(pop)), key=lambda i: -fits[i])
        elites = [pop[i] for i in ranked[:cfg.elitism]]
        next_gen = elites[:]
        while len(next_gen) < len(pop):
            p1, p2 = select_parents(pop, fits, cfg)
            c1, c2 = crossover(p1, p2, cfg.strategy.crossover)
            next_gen.append(mutate(c1, gen, cfg, num_inputs))
            if len(next_gen) < len(pop):
                next_gen.append(mutate(c2, gen, cfg, num_inputs))
        pop = next_gen

    macro_count = sum(1 for g in best if g["gate"] in MACRO_GATES)
    print(f"\nMacro usage in best genome: {macro_count}/{len(best)} gates.")
    return best, best_score, max_score

# =====================================================
# === Output Display ==================================
# =====================================================
def print_results(best, score, max_score, num_outputs, inputs, targets):
    from grad_34_simp5 import simplify_single_output
    print("\n✅ Best Network Found:", score, "/", max_score)
    for g in best:
        print(f"{g['name']}: {g['gate']}({', '.join(g['inputs'])})")

    print("\nTruth Table Check:")
    base_idx = len(best) - num_outputs
    out_names = [best[base_idx + o]["name"] for o in range(num_outputs)]
    nin = len(inputs[0])
    for idx, row in enumerate(inputs):
        inp = {chr(ord('A') + i): row[i] for i in range(nin)}
        sig = evaluate_network(best, inp)
        row_vals = [sig[name] for name in out_names]
        target_row = [targets[o][idx] for o in range(num_outputs)]
        print(f"{tuple(row)} → {row_vals} (target {target_row})")

    print("\n🧠 Simplified Output Logic:")
    for o in range(num_outputs):
        out_gate = best[-num_outputs + o]["name"]
        simp = simplify_single_output(best, out_gate)
        print(f"Output {o+1} ({out_gate}): {simp}")
