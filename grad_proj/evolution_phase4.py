# evolution_phase4.py
# Phase 4: Variable-arity gates, soft fitness, gentler mutations, local search
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Literal

from logic_gates import GATES   # same gate names as before (AND, OR, XOR, NAND, NOR, XNOR)
from grad_34_simp3 import simplify_single_output


# =============== Config ===============

@dataclass
class EvolutionStrategy:
    init: Literal["random", "hss"] = "hss"
    selection: Literal["topk", "tournament", "roulette"] = "tournament"
    crossover: Literal["one_point", "two_point", "uniform"] = "two_point"
    # mutation weights (sum doesn't need to be 1 — they are relative)
    w_replace_gate: float = 0.15         # full gate replacement (kept small)
    w_gate_type: float = 0.25            # change only gate type
    w_swap_two_inputs: float = 0.20      # swap two input wires (if >=2)
    w_rewire_one_input: float = 0.30     # rewire exactly one input
    w_change_arity: float = 0.10         # 2->3 or 3->2 and rewire as needed


@dataclass
class GAConfig:
    num_gates: int = 16                  # allow more depth for bigger I/O
    pop_size_start: int = 1000
    generations: int = 2000

    # outputs are always the last `num_outputs` gates

    # tournament/topk
    elitism: int = 8
    parent_pool_topk: int = 16
    tournament_k: int = 5

    # variable arity
    gate_min_inputs: int = 2
    gate_max_inputs: int = 3

    # mutation schedule
    base_mutation: float = 0.30
    min_mutation: float = 0.06

    # diversity
    diversity_every: int = 80
    diversity_fraction: float = 0.10

    # soft fitness
    size_penalty_per_gate: float = 0.0   # keep 0.0 to avoid hurting convergence; raise to ~0.01 later if desired

    # local search on elites each gen (gentle hill-climb)
    local_search_on_elite: bool = True
    local_search_elite_count: int = 4    # try to improve top N
    local_search_trials_per_elite: int = 2

    # module injection: copy prefix from elite into some children
    enable_module_injection: bool = True
    module_injection_rate: float = 0.20  # fraction of children to receive elite prefix
    module_prefix_len: int = 4           # number of gates copied from elite start

    # logging + rng
    log_every: int = 20
    seed: int = 42

    strategy: EvolutionStrategy = field(default_factory=EvolutionStrategy)


# =============== HSS (optional) ===============

def radical_inverse(base: int, index: int) -> float:
    inv = 0.0
    denom = 1.0
    while index > 0:
        index, rem = divmod(index, base)
        denom *= base
        inv += rem / denom
    return inv

def hammersley_point(i: int, n: int, dims: int) -> List[float]:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    point = [i / max(1, n)]
    for d in range(dims - 1):
        base = primes[d] if d < len(primes) else 2
        point.append(radical_inverse(base, i))
    return point

def hss_unit_cube(n_points: int, dims: int) -> List[List[float]]:
    return [hammersley_point(i, n_points, dims) for i in range(n_points)]


# =============== Genome helpers ===============

def _base_inputs(num_inputs: int) -> List[str]:
    return [chr(ord('A') + i) for i in range(num_inputs)]

def _available_after_index(num_inputs: int, gate_index: int) -> List[str]:
    base = _base_inputs(num_inputs)
    av = base + [f"n{x}" for x in base]
    for gi in range(gate_index):
        av.append(f"g{gi}")
    return av

def _rand_gate_inputs(available: List[str], k: int) -> List[str]:
    # allow duplicates? generally OK in logic; but avoid identical pair in 2-input for more variety
    if k == 2:
        a, b = random.sample(available, 2) if len(available) >= 2 else (available[0], available[0])
        return [a, b]
    else:
        # unique picks up to len(available); fall back to choices with replacement if needed
        if len(available) >= k:
            return random.sample(available, k)
        else:
            return [random.choice(available) for _ in range(k)]

def _rand_gate_type() -> str:
    return random.choice(list(GATES.keys()))

def random_gate(index: int, num_inputs: int, cfg: GAConfig) -> Dict[str, Any]:
    available = _available_after_index(num_inputs, index)
    k = random.randint(cfg.gate_min_inputs, cfg.gate_max_inputs)
    return {
        'name': f"g{index}",
        'gate': _rand_gate_type(),
        'inputs': _rand_gate_inputs(available, k),
    }

def random_individual(num_inputs: int, cfg: GAConfig) -> List[Dict[str, Any]]:
    indiv = []
    for i in range(cfg.num_gates):
        indiv.append(random_gate(i, num_inputs, cfg))
    return indiv

def hss_individual(num_inputs: int, cfg: GAConfig, vec: List[float]) -> List[Dict[str, Any]]:
    indiv: List[Dict[str, Any]] = []
    idx = 0
    def take():
        nonlocal idx
        v = vec[idx % len(vec)]
        idx += 1
        return v

    for gi in range(cfg.num_gates):
        available = _available_after_index(num_inputs, gi)
        k_rng = cfg.gate_max_inputs - cfg.gate_min_inputs + 1
        k = cfg.gate_min_inputs + int(take() * k_rng) % k_rng
        gate = _rand_gate_type()
        ins = []
        for _ in range(k):
            ins.append(available[int(take() * len(available)) % len(available)])
        indiv.append({'name': f"g{gi}", 'gate': gate, 'inputs': ins})
    return indiv


# =============== Simulation ===============

def _apply_gate(gname: str, in_bits: List[int]) -> int:
    # Extend 2-input gate semantics to k-input:
    # AND/OR/XOR fold across inputs; NAND/NOR/XNOR are inverted versions.
    if gname == 'AND':
        v = 1
        for b in in_bits: v &= b
        return v
    if gname == 'OR':
        v = 0
        for b in in_bits: v |= b
        return v
    if gname == 'XOR':
        v = 0
        for b in in_bits: v ^= b
        return v
    if gname == 'NAND':
        return 1 - _apply_gate('AND', in_bits)
    if gname == 'NOR':
        return 1 - _apply_gate('OR', in_bits)
    if gname == 'XNOR':
        return 1 - _apply_gate('XOR', in_bits)
    # fallback (shouldn't happen)
    return 0

def evaluate_network(individual: List[Dict[str, Any]], inputs_dict: Dict[str, int]) -> Dict[str, int]:
    signals = dict(inputs_dict)
    for k, v in list(inputs_dict.items()):
        signals[f"n{k}"] = 1 - v

    for i, gate in enumerate(individual):
        vals = [signals[name] for name in gate['inputs']]
        signals[gate['name']] = _apply_gate(gate['gate'], vals)
    return signals


# =============== Fitness ===============

def evaluate_fitness_soft(individual: List[Dict[str, Any]],
                          num_outputs: int,
                          inputs: List[Tuple[int, ...]],
                          targets: List[List[int]],
                          cfg: GAConfig) -> float:
    # Bit accuracy
    acc = 0
    base_idx = len(individual) - num_outputs
    out_names = [individual[base_idx + o]['name'] for o in range(num_outputs)]
    for row_idx, row in enumerate(inputs):
        row_map = {chr(ord('A') + i): row[i] for i in range(len(row))}
        sig = evaluate_network(individual, row_map)
        for o in range(num_outputs):
            acc += 1 if sig[out_names[o]] == targets[o][row_idx] else 0

    # optional tiny size penalty (off by default)
    penalty = cfg.size_penalty_per_gate * len(individual)
    return acc - penalty


# =============== Selection & Variation ===============

def select_parents(population: List[List[Dict[str, Any]]],
                   fitnesses: List[float],
                   cfg: GAConfig):
    sel = cfg.strategy.selection

    # --- Top-K selection ---
    if sel == "topk":
        ranked = sorted(range(len(population)), key=lambda i: -fitnesses[i])
        i1 = random.randrange(min(cfg.parent_pool_topk, len(ranked)))
        i2 = random.randrange(min(cfg.parent_pool_topk, len(ranked)))
        return population[ranked[i1]], population[ranked[i2]]

    # --- Tournament selection ---
    if sel == "tournament":
        k = max(2, cfg.tournament_k)
        def pick():
            cand = random.sample(range(len(population)), min(k, len(population)))
            best_i = max(cand, key=lambda i: fitnesses[i])
            return population[best_i]
        return pick(), pick()

    # --- Roulette selection ---
    if sel == "roulette":
        total = sum(max(0.0, f) for f in fitnesses) + 1e-9
        probs = [max(0.0, f) / total for f in fitnesses]
        def pick():
            r = random.random()
            s = 0.0
            for i, p in enumerate(probs):
                s += p
                if r <= s:
                    return population[i]
            return population[-1]
        return pick(), pick()

    # --- Fallback (random parents) ---
    return random.choice(population), random.choice(population)



def crossover(p1: List[Dict[str, Any]],
              p2: List[Dict[str, Any]],
              mode: str):
    n = len(p1)
    if n < 3: return p1[:], p2[:]
    if mode == "one_point":
        cut = random.randint(1, n - 2)
        return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]
    if mode == "two_point":
        a = random.randint(1, n - 2)
        b = random.randint(a + 1, n - 1)
        return p1[:a] + p2[a:b] + p1[b:], p2[:a] + p1[a:b] + p2[b:]
    if mode == "uniform":
        c1, c2 = [], []
        for i in range(n):
            if random.random() < 0.5:
                c1.append(p1[i].copy()); c2.append(p2[i].copy())
            else:
                c1.append(p2[i].copy()); c2.append(p1[i].copy())
        return c1, c2
    cut = random.randint(1, n - 2)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]


def _mutate_gate_shallow(g: Dict[str, Any], gi: int, num_inputs: int, cfg: GAConfig) -> Dict[str, Any]:
    # choose which op
    weights = [
        ("replace_gate", cfg.strategy.w_replace_gate),
        ("gate_type", cfg.strategy.w_gate_type),
        ("swap_inputs", cfg.strategy.w_swap_two_inputs),
        ("rewire_one", cfg.strategy.w_rewire_one_input),
        ("change_arity", cfg.strategy.w_change_arity),
    ]
    total_w = sum(w for _, w in weights)
    r = random.random() * total_w
    s = 0.0
    op = "gate_type"
    for name, w in weights:
        s += w
        if r <= s:
            op = name; break

    new_g = {'name': g['name'], 'gate': g['gate'], 'inputs': g['inputs'][:]}

    available = _available_after_index(num_inputs, gi)

    if op == "replace_gate":
        k = random.randint(cfg.gate_min_inputs, cfg.gate_max_inputs)
        new_g['gate'] = _rand_gate_type()
        new_g['inputs'] = _rand_gate_inputs(available, k)
        return new_g

    if op == "gate_type":
        new_g['gate'] = _rand_gate_type()
        return new_g

    if op == "swap_inputs" and len(new_g['inputs']) >= 2:
        a = random.randrange(len(new_g['inputs']))
        b = random.randrange(len(new_g['inputs']))
        while b == a: b = random.randrange(len(new_g['inputs']))
        new_g['inputs'][a], new_g['inputs'][b] = new_g['inputs'][b], new_g['inputs'][a]
        return new_g

    if op == "rewire_one":
        if len(new_g['inputs']) == 0:
            k = random.randint(cfg.gate_min_inputs, cfg.gate_max_inputs)
            new_g['inputs'] = _rand_gate_inputs(available, k)
        else:
            which = random.randrange(len(new_g['inputs']))
            new_g['inputs'][which] = random.choice(available)
        return new_g

    if op == "change_arity":
        k = len(new_g['inputs'])
        # adjust to new k within bounds
        if k < cfg.gate_max_inputs and random.random() < 0.5:
            # increase arity by 1
            new_g['inputs'].append(random.choice(available))
        elif k > cfg.gate_min_inputs:
            # decrease arity by 1
            drop = random.randrange(len(new_g['inputs']))
            del new_g['inputs'][drop]
        else:
            # no-op fallback
            new_g['gate'] = _rand_gate_type()
        return new_g

    return new_g


def mutate(individual: List[Dict,], gen: int, cfg: GAConfig, num_inputs: int) -> List[Dict]:
    t = gen / max(1, cfg.generations)
    rate = max(cfg.min_mutation, cfg.base_mutation * (1 - t))
    out = []
    for gi, g in enumerate(individual):
        if random.random() < rate:
            out.append(_mutate_gate_shallow(g, gi, num_inputs, cfg))
        else:
            out.append({'name': g['name'], 'gate': g['gate'], 'inputs': g['inputs'][:]})
    return out


# =============== Local Search ===============

def local_search(ind: List[Dict[str, Any]],
                 fitness: float,
                 num_inputs: int,
                 cfg: GAConfig,
                 inputs: List[Tuple[int, ...]],
                 targets: List[List[int]],
                 num_outputs: int) -> Tuple[List[Dict[str, Any]], float]:
    # Try a few single-gate tweaks; accept if improves
    best = ind
    best_f = fitness
    trials = cfg.local_search_trials_per_elite
    for _ in range(trials):
        gi = random.randrange(len(ind))
        cand = ind[:]
        cand[gi] = _mutate_gate_shallow(ind[gi], gi, num_inputs, cfg)
        f = evaluate_fitness_soft(cand, num_outputs, inputs, targets, cfg)
        if f > best_f:
            best, best_f = cand, f
    return best, best_f


# =============== Evolution ===============

def evolve_phase4(num_inputs: int,
                  num_outputs: int,
                  inputs: List[Tuple[int, ...]],
                  targets: List[List[int]],
                  cfg: GAConfig):
    random.seed(cfg.seed)

    # init
    pop = []
    if cfg.strategy.init == "hss":
        dims = max(8, cfg.num_gates * 3)
        for vec in hss_unit_cube(cfg.pop_size_start, dims):
            pop.append(hss_individual(num_inputs, cfg, vec))
    else:
        pop = [random_individual(num_inputs, cfg) for _ in range(cfg.pop_size_start)]

    max_bits = len(inputs) * num_outputs
    history = {'gen': [], 'best': [], 'pop': []}
    best, best_f = None, -1e9

    for gen in range(cfg.generations):
        # fitness
        fits = [evaluate_fitness_soft(ind, num_outputs, inputs, targets, cfg) for ind in pop]
        ranked_idx = sorted(range(len(pop)), key=lambda i: -fits[i])
        cur_best = pop[ranked_idx[0]]
        cur_best_f = fits[ranked_idx[0]]

        if cur_best_f > best_f:
            best, best_f = cur_best, cur_best_f

        if gen % cfg.log_every == 0 or int(best_f) >= max_bits:
            # Since we used soft fitness (float), print int(best_f)/max_bits
            print(f"Gen {gen:4d} | Best {int(best_f)}/{max_bits} | Pop {len(pop)}")

        history['gen'].append(gen)
        history['best'].append(int(best_f))
        history['pop'].append(len(pop))

        if int(best_f) >= max_bits:  # perfect accuracy achieved
            break

        # diversity injection
        if cfg.diversity_every > 0 and gen > 0 and gen % cfg.diversity_every == 0:
            inject = max(1, int(len(pop) * cfg.diversity_fraction))
            for i in range(inject):
                if cfg.strategy.init == "hss":
                    dims = max(8, cfg.num_gates * 3)
                    vec = hammersley_point(gen * inject + i + 1, max(1, len(pop)), dims)
                    pop[-(i+1)] = hss_individual(num_inputs, cfg, vec)
                else:
                    pop[-(i+1)] = random_individual(num_inputs, cfg)

        # elites (and optional local search)
        elites = [pop[i] for i in ranked_idx[:cfg.elitism]]
        elite_f = [fits[i] for i in ranked_idx[:cfg.elitism]]
        if cfg.local_search_on_elite:
            for k in range(min(cfg.local_search_elite_count, len(elites))):
                improved, imp_f = local_search(elites[k], elite_f[k], num_inputs, cfg, inputs, targets, num_outputs)
                if imp_f > elite_f[k]:
                    elites[k] = improved
                    elite_f[k] = imp_f
                    if imp_f > best_f: best, best_f = improved, imp_f

        # next gen
        next_pop = elites[:]
        while len(next_pop) < len(pop):
            p1, p2 = select_parents(pop, fits, cfg)
            c1, c2 = crossover(p1, p2, cfg.strategy.crossover)

            # module injection: copy prefix from best into some children
            if cfg.enable_module_injection and random.random() < cfg.module_injection_rate:
                pref = cfg.module_prefix_len
                c1[:pref] = best[:pref]
            if cfg.enable_module_injection and random.random() < cfg.module_injection_rate:
                pref = cfg.module_prefix_len
                c2[:pref] = best[:pref]

            next_pop.append(mutate(c1, gen, cfg, num_inputs))
            if len(next_pop) < len(pop):
                next_pop.append(mutate(c2, gen, cfg, num_inputs))
        pop = next_pop

    return best, int(best_f), max_bits, history


# =============== Pretty print ===============

def print_results(best: List[Dict[str, Any]],
                  score_bits: int,
                  max_bits: int,
                  num_outputs: int,
                  inputs: List[Tuple[int, ...]],
                  targets: List[List[int]]) -> None:

    print("\n✅ Best Network Found:", score_bits, "/", max_bits)
    print("GATE LIST (variable arity):")
    for g in best:
        ins = ",".join(g['inputs'])
        print(f"{g['name']}: {g['gate']}({ins})")

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
