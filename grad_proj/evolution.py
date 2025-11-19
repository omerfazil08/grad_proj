# evolution.py
# -----------------
import random
import copy
from functools import lru_cache
from logic_gates import GATES
from grad_34_simp2 import simplify_single_output


# --- Evaluation of a network ---
def evaluate_network(individual, inputs_dict):
    """Simulate the circuit for one input pattern."""
    signals = {**inputs_dict, **{f'n{k}': 1-v for k, v in inputs_dict.items()}}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals


# Cache network evaluations for speed
@lru_cache(maxsize=None)
def evaluate_network_cached(individual_repr, inputs_tuple):
    """Cached version using hashable representations."""
    import ast
    individual = ast.literal_eval(individual_repr)
    inputs_dict = {chr(65+i): bit for i, bit in enumerate(inputs_tuple)}
    return evaluate_network(individual, inputs_dict)


# --- Fitness evaluation ---
def evaluate_fitness(individual, inputs, targets, num_outputs, penalty=0.05):
    """Calculate normalized fitness between 0 and 1."""
    correct = 0
    max_score = len(inputs) * num_outputs
    individual_repr = repr(individual)

    for row_idx, inputs_tuple in enumerate(inputs):
        outputs = evaluate_network_cached(individual_repr, inputs_tuple)
        for o in range(num_outputs):
            gate_name = individual[-num_outputs + o]['name']
            if outputs[gate_name] == targets[o][row_idx]:
                correct += 1

    # Add light penalty for complexity
    fitness = (correct / max_score) - penalty * (len(individual) / 10)
    return max(fitness, 0.0)


# --- Circuit generation ---
def random_gate(index, available):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available, 2)
    }


def random_individual(num_inputs=3, num_gates=8):
    individual = []
    base_inputs = [chr(65 + i) for i in range(num_inputs)]  # A, B, C...
    available = base_inputs + [f'n{k}' for k in base_inputs]
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual


# --- Mutation ---
def mutate_gate(gate, available):
    new_gate = copy.deepcopy(gate)
    choice = random.choice(["type", "inputs", "both"])
    if choice in ["type", "both"]:
        new_gate["gate"] = random.choice(list(GATES.keys()))
    if choice in ["inputs", "both"]:
        new_gate["inputs"] = random.sample(available, 2)
    return new_gate


def mutate(individual, gen, generations, base_rate=0.3):
    mutation_rate = base_rate * (1 - gen / generations)
    mutant = []
    base_inputs = ['A', 'B', 'C']
    available = base_inputs + [f'n{k}' for k in base_inputs]
    for gate in individual:
        if random.random() < mutation_rate:
            new_gate = mutate_gate(gate, available)
        else:
            new_gate = copy.deepcopy(gate)
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant


# --- Crossover ---
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 2)
    c1 = copy.deepcopy(p1[:point] + p2[point:])
    c2 = copy.deepcopy(p2[:point] + p1[point:])
    return c1, c2


# --- Parent selection (tournament) ---
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]


# --- Evolution main loop ---
def evolve(num_inputs, num_outputs, inputs, targets, pop_size=150, generations=700, num_gates=8):
    random.seed(42)  # reproducibility

    population = [random_individual(num_inputs, num_gates) for _ in range(pop_size)]
    max_fitness = 1.0

    for gen in range(generations):
        fitnesses = [evaluate_fitness(ind, inputs, targets, num_outputs) for ind in population]
        ranked = sorted(zip(population, fitnesses), key=lambda x: -x[1])
        best, best_fit = ranked[0]

        if gen % 50 == 0 or best_fit >= max_fitness:
            print(f"Generation {gen:3d} | Best Fitness: {best_fit:.3f}")

        if best_fit >= max_fitness:
            print("✅ Perfect match found!")
            break

        # Elitism
        next_gen = [copy.deepcopy(best)]

        # Generate next generation
        while len(next_gen) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            next_gen.append(mutate(child1, gen, generations))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(child2, gen, generations))

        population = next_gen

    return best, best_fit


# --- Truth table testing ---
def print_truth_table(best, num_inputs, num_outputs, inputs, targets):
    print("\nTruth Table Check:")
    for idx, input_tuple in enumerate(inputs):
        inputs_dict = {chr(65 + i): v for i, v in enumerate(input_tuple)}
        signals = evaluate_network(best, inputs_dict)
        row_out = [signals[best[-num_outputs + o]['name']] for o in range(num_outputs)]
        print(f"{input_tuple} → {row_out}   (target {[targets[o][idx] for o in range(num_outputs)]})")
