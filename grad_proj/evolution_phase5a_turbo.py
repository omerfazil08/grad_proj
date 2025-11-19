# evolution_phase5a_turbo.py
# Phase 5a — Turbo: macros + (diversity, elite-local-search, soft fitness, weighted mutation)
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Literal, Optional

# =========================
# Gate library (with macros)
# =========================
def AND(*xs):  # var-arity (2..3)
    v = 1
    for x in xs[:3]:
        v &= x
    return v

def OR(*xs):  # var-arity (2..3)
    v = 0
    for x in xs[:3]:
        v |= x
    return v

def XOR2(a, b):  return a ^ b
def XNOR2(a, b): return 1 - (a ^ b)
def NAND(*xs):   return 1 - AND(*xs)
def NOR(*xs):    return 1 - OR(*xs)

# ---- Macros (fixed arity) ----
def HALF_SUM(a, b):      return a ^ b                         # a ⊕ b
def HALF_CARRY(a, b):    return a & b                         # a · b
def FULL_SUM(a, b, c):   return (a ^ b) ^ c                   # a ⊕ b ⊕ c
def FULL_CARRY(a, b, c): return (a & b) | (a & c) | (b & c)   # Σ2
# IMPORTANT: fixed order (a, b, s)  -> (~s & a) | (s & b)
def MUX2(a, b, s):       return ((1 - s) & a) | (s & b)
def EQ1(a, b):           return 1 - (a ^ b)                   # XNOR
def GT1(a, b):           return a & (1 - b)                   # a > b
def LT1(a, b):           return (1 - a) & b                   # a < b

PRIMITIVES = ["AND", "OR", "NAND", "NOR", "XOR2", "XNOR2"]
MACROS     = ["HALF_SUM", "HALF_CARRY", "FULL_SUM", "FULL_CARRY", "MUX2", "EQ1", "GT1", "LT1"]

GATE_FNS = {
    "AND": AND, "OR": OR, "NAND": NAND, "NOR": NOR, "XOR2": XOR2, "XNOR2": XNOR2,
    "HALF_SUM": HALF_SUM, "HALF_CARRY": HALF_CARRY,
    "FULL_SUM": FULL_SUM, "FULL_CARRY": FULL_CARRY,
    "MUX2": MUX2, "EQ1": EQ1, "GT1": GT1, "LT1": LT1,
}

# For fixed-arity macros; primitives are "var" (2..3)
GATE_ARITY: Dict[str, Any] = {
    "AND": "var", "OR": "var", "NAND": "var", "NOR": "var",
    "XOR2": 2, "XNOR2": 2,
    "HALF_SUM": 2, "HALF_CARRY": 2,
    "FULL_SUM": 3, "FULL_CARRY": 3,
    "MUX2": 3, "EQ1": 2, "GT1": 2, "LT1": 2,
}

# =========================
# Config
# =========================
@dataclass
class TurboConfig:
    num_gates: int = 16
    pop_size: int = 800
    generations: int = 1200

    # Selection / elitism
    elitism: int = 10
    tournament_k: int = 5

    # Mutation schedule
    base_mut: float = 0.30
    min_mut: float = 0.06

    # Macro usage control (favor primitives to stabilize search)
    p_choose_primitive: float = 0.70   # when replacing a gate, 70% primitives / 30% macros

    # Diversity injection
    diversity_every: int = 80
    diversity_fraction: float = 0.10

    # Elite local search
    local_search_on_elite: bool = True
    local_search_elites: int = 6
    local_search_trials: int = 2

    # Logging
    log_every: int = 20

    # Reproducibility
    seed: int = 42

    # Optional size penalty (kept tiny; 0 disables)
    size_penalty_lambda: float = 0.0


# =========================
# Helpers
# =========================
def base_inputs(n: int) -> List[str]:
    letters = [chr(ord('A') + i) for i in range(n)]
    return letters

def available_after_index(n_inputs: int, idx: int) -> List[str]:
    av = base_inputs(n_inputs) + [f"n{c}" for c in base_inputs(n_inputs)]
    for gi in range(idx):
        av.append(f"g{gi}")
    return av

def hammersley_point(i: int, n: int, dims: int) -> List[float]:
    # i/n plus van der Corput for remaining dims
    def radical_inverse(base: int, index: int) -> float:
        inv, denom = 0.0, 1.0
        while index > 0:
            index, rem = divmod(index, base)
            denom *= base
            inv += rem / denom
        return inv
    primes = [2,3,5,7,11,13,17,19]
    pt = [i / max(1, n)]
    for d in range(dims - 1):
        b = primes[d % len(primes)]
        pt.append(radical_inverse(b, i))
    return pt

def hss_unit_cube(n_points: int, dims: int) -> List[List[float]]:
    return [hammersley_point(i, n_points, dims) for i in range(n_points)]

def choose_gate_name(p_prim: float) -> str:
    if random.random() < p_prim:
        return random.choice(PRIMITIVES)
    return random.choice(MACROS)

# =========================
# Genome creation / mutation
# =========================
def random_gate(idx: int, n_inputs: int, p_prim: float) -> Dict[str, Any]:
    gname = choose_gate_name(p_prim)
    ar = GATE_ARITY[gname]
    av = available_after_index(n_inputs, idx)
    if ar == "var":
        # pick 2 or 3 inputs
        k = 2 if random.random() < 0.7 else 3
        ins = random.sample(av, k)
    else:
        ins = random.sample(av, ar)
    return {"name": f"g{idx}", "gate": gname, "inputs": ins}

def random_individual(n_inputs: int, n_gates: int, p_prim: float) -> List[Dict[str, Any]]:
    indiv = []
    for i in range(n_gates):
        indiv.append(random_gate(i, n_inputs, p_prim))
    return indiv

def hss_individual(n_inputs: int, n_gates: int, vec: List[float], p_prim: float) -> List[Dict[str, Any]]:
    indiv = []
    idx = 0
    def take():
        nonlocal idx
        v = vec[idx % len(vec)]
        idx += 1
        return v
    for gi in range(n_gates):
        gtype = choose_gate_name(p_prim)
        av = available_after_index(n_inputs, gi)
        ar = GATE_ARITY[gtype]
        if ar == "var":
            k = 2 if take() < 0.7 else 3
            a = int(take() * len(av)) % len(av)
            b = int(take() * len(av)) % len(av)
            ins = [av[a], av[b]]
            if k == 3:
                c = int(take() * len(av)) % len(av)
                ins.append(av[c])
        else:
            ins = []
            for _ in range(ar):
                ins.append(av[int(take() * len(av)) % len(av)])
        indiv.append({"name": f"g{gi}", "gate": gtype, "inputs": ins})
    return indiv

def mutate_gate(g: Dict[str,Any], gate_index: int, n_inputs: int, cfg: TurboConfig) -> Dict[str,Any]:
    # Small, safe edits
    newg = dict(g)
    r = random.random()
    # 40% rewire one input
    if r < 0.40:
        av = available_after_index(n_inputs, gate_index)
        which = random.randrange(len(newg["inputs"]))
        newg["inputs"][which] = random.choice(av)
        return newg
    r -= 0.40
    # 25% swap inputs (if >=2)
    if r < 0.25 and len(newg["inputs"]) >= 2:
        a, b = newg["inputs"][0], newg["inputs"][1]
        newg["inputs"][0], newg["inputs"][1] = b, a
        return newg
    r -= 0.25
    # 20% change gate type (keep arity reasonable)
    if r < 0.20:
        gname = choose_gate_name(cfg.p_choose_primitive)
        ar = GATE_ARITY[gname]
        av = available_after_index(n_inputs, gate_index)
        if ar == "var":
            k = 2 if random.random() < 0.7 else 3
            ins = random.sample(av, k)
        else:
            ins = random.sample(av, ar)
        newg["gate"] = gname
        newg["inputs"] = ins
        return newg
    # else: full replacement
    av = available_after_index(n_inputs, gate_index)
    gname = choose_gate_name(cfg.p_choose_primitive)
    ar = GATE_ARITY[gname]
    if ar == "var":
        k = 2 if random.random() < 0.7 else 3
        ins = random.sample(av, k)
    else:
        ins = random.sample(av, ar)
    return {"name": g["name"], "gate": gname, "inputs": ins}

def mutate(ind: List[Dict[str,Any]], gen: int, cfg: TurboConfig, n_inputs: int) -> List[Dict[str,Any]]:
    t = gen / max(1, cfg.generations)
    rate = max(cfg.min_mut, cfg.base_mut * (1 - 0.7 * t))
    out = []
    for i, g in enumerate(ind):
        if random.random() < rate:
            out.append(mutate_gate(g, i, n_inputs, cfg))
        else:
            out.append(dict(g))
    return out

def crossover(p1: List[Dict[str,Any]], p2: List[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    n = len(p1)
    if n < 3:
        return [dict(x) for x in p1], [dict(x) for x in p2]
    a = random.randint(1, n-2)
    b = random.randint(a+1, n-1)
    c1 = [dict(x) for x in (p1[:a] + p2[a:b] + p1[b:])]
    c2 = [dict(x) for x in (p2[:a] + p1[a:b] + p2[b:])]
    return c1, c2

# =========================
# Evaluation
# =========================
def eval_gate(name: str, ins: List[int]) -> int:
    fn = GATE_FNS[name]
    ar = GATE_ARITY[name]
    if ar == "var":
        if len(ins) < 2:
            ins = ins * 2
        return int(fn(*ins[:3]))
    elif ar == 2:
        if len(ins) < 2:
            ins = ins + ins
        return int(fn(ins[0], ins[1]))
    else:  # ar == 3
        if len(ins) < 3:
            ins = ins + [ins[-1]] * (3 - len(ins))
        return int(fn(ins[0], ins[1], ins[2]))

def evaluate_network(ind: List[Dict[str,Any]], input_dict: Dict[str,int]) -> Dict[str,int]:
    signals = {}
    # set base + complements
    for k, v in input_dict.items():
        signals[k] = v
        signals[f"n{k}"] = 1 - v
    # forward eval
    for g in ind:
        in_vals = [signals[x] for x in g["inputs"]]
        signals[g["name"]] = eval_gate(g["gate"], in_vals)
    return signals

def soft_fitness(ind: List[Dict[str,Any]],
                 n_outputs: int,
                 inputs: List[Tuple[int,...]],
                 targets: List[List[int]]) -> int:
    """
    Soft score = number of matching bits across all rows & outputs (same as Hamming matches).
    This is 'soft' compared to a single 0/1 per row; for multi-output it gives smoother gradient.
    """
    score = 0
    base_idx = len(ind) - n_outputs
    out_names = [ind[base_idx + o]["name"] for o in range(n_outputs)]
    cache: Dict[Tuple[int,...], List[int]] = {}  # small per-call cache

    for r, row in enumerate(inputs):
        if row in cache:
            outs = cache[row]
        else:
            inp = {chr(ord('A') + i): row[i] for i in range(len(row))}
            sig = evaluate_network(ind, inp)
            outs = [sig[name] for name in out_names]
            cache[row] = outs
        for o in range(n_outputs):
            if outs[o] == targets[o][r]:
                score += 1
    return score

# =========================
# Selection
# =========================
def tournament_select(pop: List[List[Dict[str,Any]]], fits: List[int], k: int) -> List[Dict[str,Any]]:
    cand_idx = random.sample(range(len(pop)), min(k, len(pop)))
    best_i = max(cand_idx, key=lambda i: fits[i])
    return pop[best_i]

# =========================
# Evolution loop
# =========================
def evolve_5a_turbo(n_inputs: int,
                    n_outputs: int,
                    inputs: List[Tuple[int,...]],
                    targets: List[List[int]],
                    cfg: TurboConfig):
    random.seed(cfg.seed)
    max_score = len(inputs) * n_outputs

    # HSS init
    dims = max(6, 3 * cfg.num_gates)
    hss = hss_unit_cube(cfg.pop_size, dims)
    population = [hss_individual(n_inputs, cfg.num_gates, hss[i], cfg.p_choose_primitive)
                  for i in range(cfg.pop_size)]

    best, best_score = None, -1

    for gen in range(cfg.generations + 1):
        fits = [soft_fitness(ind, n_outputs, inputs, targets) for ind in population]

        # Rank and record
        ranked = sorted(zip(population, fits), key=lambda x: -x[1])
        cur_best, cur_score = ranked[0]
        if cur_score > best_score:
            best, best_score = cur_best, cur_score

        if (gen % cfg.log_every == 0) or (cur_score == max_score):
            print(f"Gen {gen:4d} | Best {cur_score}/{max_score} | Pop {len(population)}")

        if cur_score == max_score:
            break

        # Diversity injection
        if cfg.diversity_every and gen and (gen % cfg.diversity_every == 0):
            inject = max(1, int(cfg.pop_size * cfg.diversity_fraction))
            for i in range(inject):
                vec = hammersley_point(gen * inject + i + 1, max(1, cfg.pop_size), dims)
                population[-(i+1)] = hss_individual(n_inputs, cfg.num_gates, vec, cfg.p_choose_primitive)

        # Next gen with elitism
        elites = [p for p, _ in ranked[:cfg.elitism]]
        next_gen = [ [dict(g) for g in e] for e in elites ]

        # Elite local search (polish)
        if cfg.local_search_on_elite:
            for ei in range(min(cfg.local_search_elites, len(next_gen))):
                base = next_gen[ei]
                best_local = base
                best_local_fit = soft_fitness(best_local, n_outputs, inputs, targets)
                for _ in range(cfg.local_search_trials):
                    cand = mutate(base, gen, cfg, n_inputs)
                    f = soft_fitness(cand, n_outputs, inputs, targets)
                    if f > best_local_fit:
                        best_local, best_local_fit = cand, f
                next_gen[ei] = best_local

        # Fill rest
        while len(next_gen) < cfg.pop_size:
            p1 = tournament_select(population, fits, cfg.tournament_k)
            p2 = tournament_select(population, fits, cfg.tournament_k)
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1, gen, cfg, n_inputs))
            if len(next_gen) < cfg.pop_size:
                next_gen.append(mutate(c2, gen, cfg, n_inputs))

        population = next_gen

    return best, best_score, max_score

# =========================
# Pretty print & check
# =========================
def print_results(best: List[Dict[str,Any]],
                  score: int,
                  max_score: int,
                  n_outputs: int,
                  inputs: List[Tuple[int,...]],
                  targets: List[List[int]]):
    print(f"\n✅ Best Network Found: {score} / {max_score}")
    print("GATE LIST (turbo):")
    for g in best:
        ins = ",".join(g["inputs"])
        print(f"{g['name']}: {g['gate']}({ins})")

    base_idx = len(best) - n_outputs
    out_names = [best[base_idx + o]["name"] for o in range(n_outputs)]

    print("\nTruth Table Check:")
    for r, row in enumerate(inputs):
        inp = {chr(ord('A') + i): row[i] for i in range(len(row))}
        sig = evaluate_network(best, inp)
        outs = [sig[name] for name in out_names]
        print(f"{tuple(row)} → {outs}   (target {[targets[o][r] for o in range(n_outputs)]})")
