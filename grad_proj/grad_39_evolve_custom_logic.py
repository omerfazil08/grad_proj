import random
import copy

# --- Basic logic gates ---
def AND(a, b): return a & b
def OR(a, b): return a | b
def XOR(a, b): return a ^ b
def NAND(a, b): return ~(a & b) & 1
def NOR(a, b): return ~(a | b) & 1
def XNOR(a, b): return ~(a ^ b) & 1

GATES = {
    'AND': AND,
    'OR': OR,
    'XOR': XOR,
    'NAND': NAND,
    'NOR': NOR,
    'XNOR': XNOR
}

# -------------------------
# Utilities
# -------------------------
def build_input_names(n):
    return [chr(ord('A') + i) for i in range(n)]

def neg_name(name):
    return f"n{name}"

def random_gate(index, available_inputs):
    return {
        'name': f"g{index}",
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

# -------------------------
# Individual creation
# -------------------------
def random_individual(num_inputs, num_outputs, num_gates=8):
    input_names = build_input_names(num_inputs)
    available = input_names[:] + [neg_name(n) for n in input_names]
    gates = []
    for i in range(num_gates):
        g = random_gate(i, available)
        gates.append(g)
        available.append(g['name'])
    outputs = random.sample(available, num_outputs)
    return {'gates': gates, 'outputs': outputs}

# -------------------------
# Evaluation
# -------------------------
def evaluate_network(individual, input_values):
    signals = {}
    for k, v in input_values.items():
        signals[k] = v
        signals[neg_name(k)] = 1 - v
    for gate in individual['gates']:
        in1 = signals.get(gate['inputs'][0], 0)
        in2 = signals.get(gate['inputs'][1], 0)
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

def evaluate_fitness(individual, input_names, inputs_list, targets):
    score = 0
    for idx, vec in enumerate(inputs_list):
        iv = {name: bit for name, bit in zip(input_names, vec)}
        signals = evaluate_network(individual, iv)
        for o_index, out_name in enumerate(individual['outputs']):
            bit = signals.get(out_name, 0)
            if bit == targets[o_index][idx]:
                score += 1
    return score

# -------------------------
# Mutation (with adaptive rate)
# -------------------------
def mutate(individual, input_names, gen, generations, base_mutation=0.3):
    """
    Adaptive mutation rate decreases linearly from base_mutation → 0 as gen increases.
    """
    mutant = copy.deepcopy(individual)
    available = input_names[:] + [neg_name(n) for n in input_names] + [g['name'] for g in mutant['gates']]
    mutation_rate = base_mutation * (1 - gen / generations)

    for i, gate in enumerate(mutant['gates']):
        if random.random() < mutation_rate:
            mutant['gates'][i] = random_gate(i, available)
        available.append(mutant['gates'][i]['name'])

    # also small chance to change output
    for i in range(len(mutant['outputs'])):
        if random.random() < mutation_rate * 0.5:
            mutant['outputs'][i] = random.choice(available)

    return mutant

# -------------------------
# Evolution loop
# -------------------------
def evolve(num_inputs, num_outputs, inputs_list, targets,
           pop_size=100, generations=400, num_gates=8, base_mutation=0.3, elitism=4):
    input_names = build_input_names(num_inputs)
    population = [random_individual(num_inputs, num_outputs, num_gates) for _ in range(pop_size)]
    max_score = len(inputs_list) * num_outputs
    best = None
    best_score = -1

    for gen in range(generations):
        fitness_list = [evaluate_fitness(ind, input_names, inputs_list, targets) for ind in population]
        paired = list(zip(population, fitness_list))
        paired.sort(key=lambda x: x[1], reverse=True)
        best_ind, best_fit = paired[0]

        if best_fit > best_score:
            best = copy.deepcopy(best_ind)
            best_score = best_fit

        if gen % 25 == 0 or best_fit == max_score:
            print(f"Gen {gen:3d}: best = {best_fit}/{max_score}")

        if best_fit == max_score:
            print("✅ Perfect solution found!")
            break

        # elitism: carry top few
        new_pop = [copy.deepcopy(ind) for ind, _ in paired[:elitism]]

        # fill rest
        while len(new_pop) < pop_size:
            parent = random.choice(paired[:10])[0]
            child = mutate(parent, input_names, gen, generations, base_mutation)
            new_pop.append(child)

        population = new_pop

    return best, best_score, max_score, input_names

# -------------------------
# Helpers
# -------------------------
def all_input_combinations(num_inputs):
    combos = []
    for i in range(2 ** num_inputs):
        bits = tuple((i >> j) & 1 for j in reversed(range(num_inputs)))
        combos.append(bits)
    return combos

# -------------------------
# Example run
# -------------------------
if __name__ == "__main__":
    print("=== grad_37_mutrate.py ===")
    # Example: Half-adder (2 inputs, 2 outputs)
    num_inputs = 2
    num_outputs = 2
    input_names = build_input_names(num_inputs)
    inputs_list = all_input_combinations(num_inputs)

    # Half adder truth table
    # SUM = A XOR B
    # CARRY = A AND B
    targets = [
        [a ^ b for a, b in inputs_list],  # SUM
        [a & b for a, b in inputs_list]   # CARRY
    ]

    best, score, max_score, input_names = evolve(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        inputs_list=inputs_list,
        targets=targets,
        pop_size=120,
        generations=400,
        num_gates=8,
        base_mutation=0.3,
        elitism=4
    )

    print("\n✅ Best score:", score, "/", max_score)
    print("Gates:")
    for g in best['gates']:
        print(f"{g['name']}: {g['gate']}({g['inputs'][0]}, {g['inputs'][1]})")
    print("Outputs:", best['outputs'])
