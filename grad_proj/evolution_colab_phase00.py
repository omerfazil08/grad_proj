import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Callable


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
        seed: int,
        size_penalty_lambda: float,
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


# --- Global Gate Set (for convenience) ---
GATES_SET = {
    "AND": {"func": AND, "arity": "variable", "min_arity": 2},
    "OR": {"func": OR, "arity": "variable", "min_arity": 2},
    "NOT": {"func": NOT, "arity": 1},
    "XOR2": {"func": XOR2, "arity": 2},
    "XNOR2": {"func": XNOR2, "arity": 2},
    "NAND": {"func": NAND, "arity": "variable", "min_arity": 2},
    "NOR": {"func": NOR, "arity": "variable", "min_arity": 2},
    "EQ1": {"func": EQ1, "arity": 2},
    "MUX2": {"func": MUX2, "arity": 3},
    "HALF_SUM": {"func": HALF_SUM, "arity": 2},
    "HALF_CARRY": {"func": HALF_CARRY, "arity": 2},
    "FULL_SUM": {"func": FULL_SUM, "arity": 3},
    "FULL_CARRY": {"func": FULL_CARRY, "arity": 3},
}

# --- Individual (Circuit) Representation ---
def gate_name(idx):
    return f"G{idx}"

def random_gate(num_inputs: int, idx: int, cfg: ColabConfig) -> Dict[str, Any]:
    srcs = allowed_sources(num_inputs, idx)

    # Filter gates that can actually be formed with the available sources
    possible_gate_types = []
    for gate_type, gate_info in GATES_SET.items():
        if gate_info["arity"] == "variable":
            if len(srcs) >= gate_info["min_arity"]:
                possible_gate_types.append(gate_type)
        else:  # Fixed arity
            if len(srcs) >= gate_info["arity"]:
                possible_gate_types.append(gate_type)

    # If no gates can be formed (e.g., no inputs yet), create a dummy gate or handle gracefully
    if not possible_gate_types:
        # Fallback: create a NOT gate if at least one source is available
        if srcs:
            gate_type = "NOT"
            k = 1
            ins = random.sample(srcs, k)
        else: # Truly no sources, return a simple AND gate with dummy inputs (will effectively be 0)
            gate_type = "AND"
            k = 2
            ins = [f"A{i % num_inputs}" for i in range(k)] if num_inputs > 0 else ["0", "0"]
            print(f"Warning: No valid sources for gate {idx}. Using dummy inputs for {gate_type}.")
    else:
        gate_type = random.choice(possible_gate_types)
        gate_info = GATES_SET[gate_type]

        if gate_info["arity"] == "variable":
            # Choose k between min_arity and 3 (max for these variable gates), but not more than available sources
            k = random.randint(gate_info["min_arity"], min(3, len(srcs)))
        else:
            k = int(gate_info["arity"])

        ins = random.sample(srcs, k)

    return {
        "name": gate_name(idx),
        "type": gate_type,
        "inputs": ins,
        "output": gate_name(idx), # Output is just the gate's name for consistency
    }

def random_individual(num_inputs: int, cfg: ColabConfig) -> List[Dict[str, Any]]:
    return [random_gate(num_inputs, i, cfg) for i in range(cfg.num_gates)]

# --- Evaluation ---
def allowed_sources(num_inputs: int, current_gate_idx: int) -> List[str]:
    sources = [f"A{i}" for i in range(num_inputs)] # Primary inputs A0, A1, ...
    sources.extend([gate_name(i) for i in range(current_gate_idx)]) # Outputs of previous gates
    return sources

def evaluate_network(individual: List[Dict[str, Any]], inputs: Tuple[int, ...]) -> Dict[str, int]:
    # Mapping from source name (e.g., A0, G0) to its evaluated value
    values = {f"A{i}": val for i, val in enumerate(inputs)}

    for gate in individual:
        gate_type = gate["type"]
        gate_func = GATES_SET[gate_type]["func"]
        gate_inputs = gate["inputs"]
        gate_output_name = gate["output"]

        # Resolve input values for the current gate
        try:
            resolved_inputs = [values[src] for src in gate_inputs]
        except KeyError as e:
            # This should ideally not happen with the corrected random_gate, but for robustness:
            print(f"Warning: Missing input {e} for gate {gate_output_name}. Assigning 0.")
            # Assign a default value (e.g., 0) if an input is unexpectedly missing
            # This can happen if random_gate creates an invalid connection in rare edge cases
            values[gate_output_name] = 0
            continue

        values[gate_output_name] = gate_func(*resolved_inputs)
    return values

# --- Fitness Calculation ---
def fitness(individual: List[Dict[str, Any]], num_inputs: int, num_outputs: int,
            inputs_set: List[Tuple[int, ...]], targets_set: List[List[int]], cfg: ColabConfig) -> float:

    score = 0
    # Ensure at least 'num_outputs' gates exist, if not, penalize heavily or exit early.
    # This is a simplification; a full GA would evolve outputs more explicitly.
    if len(individual) < num_outputs:
        return -1.0 * (cfg.generations + 1) # Very low score

    # The outputs are assumed to be the last 'num_outputs' gates in the individual,
    # or explicitly defined output gates if the representation allowed for it.
    # For this simplified model, we'll take the last N gates as outputs.
    output_gate_names = [gate["output"] for gate in individual[-num_outputs:]]

    for i, input_tuple in enumerate(inputs_set):
        evaluated_values = evaluate_network(individual, input_tuple)

        # Extract the values for the designated output gates
        circuit_outputs = [
            evaluated_values.get(name, 0) for name in output_gate_names
        ]

        # Compare with target outputs
        for j in range(num_outputs):
            if j < len(targets_set) and i < len(targets_set[j]):
                if circuit_outputs[j] == targets_set[j][i]:
                    score += 1

    # Optional: penalize for circuit size or complexity
    size_penalty = cfg.size_penalty_lambda * len(individual)
    return score - size_penalty

# --- Selection, Crossover, Mutation ---
def select_parent_tournament(population: List[List[Dict[str, Any]]],
                             fitness_scores: List[float], cfg: ColabConfig) -> List[Dict[str, Any]]:
    tournament_members_indices = random.sample(range(len(population)), cfg.tournament_k)
    tournament_scores = [fitness_scores[i] for i in tournament_members_indices]
    winner_index = tournament_members_indices[tournament_scores.index(max(tournament_scores))]
    return population[winner_index]

def crossover(parent1: List[Dict[str, Any]], parent2: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # One-point crossover
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual: List[Dict[str, Any]], num_inputs: int, mut_rate: float, cfg: ColabConfig) -> List[Dict[str, Any]]:
    mutated_individual = []
    for i, gate in enumerate(individual):
        if random.random() < mut_rate:
            # Mutate this gate: replace with a new random gate
            mutated_individual.append(random_gate(num_inputs, i, cfg))
        else:
            mutated_individual.append(gate)
    return mutated_individual

# --- Main Evolutionary Loop ---
def evolve_colab_phase0(
    num_inputs: int,
    num_outputs: int,
    inputs_set: List[Tuple[int, ...]],
    targets_set: List[List[int]],
    cfg: ColabConfig,
) -> Tuple[List[Dict[str, Any]], float, float, Dict[str, List[Any]]]:

    random.seed(cfg.seed)

    population = [random_individual(num_inputs, cfg) for _ in range(cfg.pop_size)]
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

        new_population = []

        # Elitism: carry over the best individuals
        elite_indices = sorted(range(len(population)), key=lambda k: fitness_scores[k], reverse=True)[:cfg.elitism]
        for i in elite_indices:
            new_population.append(population[i])

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < cfg.pop_size:
            parent1 = select_parent_tournament(population, fitness_scores, cfg)
            parent2 = select_parent_tournament(population, fitness_scores, cfg)

            child1, child2 = crossover(parent1, parent2)

            # Adaptive mutation rate (example)
            mutation_rate = cfg.base_mut - (cfg.base_mut - cfg.min_mut) * (gen / cfg.generations)
            mutation_rate = max(cfg.min_mut, mutation_rate)

            new_population.append(mutate(child1, num_inputs, mutation_rate, cfg))
            if len(new_population) < cfg.pop_size:
                new_population.append(mutate(child2, num_inputs, mutation_rate, cfg))

        population = new_population

    # Final evaluation of the best individual
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
