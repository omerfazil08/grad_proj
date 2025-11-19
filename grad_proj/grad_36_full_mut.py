import random
from grad_34_simp2 import simplify_single_output

# --- Logic Gates ---
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

# --- Target Logic ---
def full_adder_sum(a, b, c): return a ^ b ^ c
def full_adder_carry(a, b, c): return (a & b) | (c & (a ^ b))

def generate_inputs():
    return [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]

# --- Create random gate ---
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

# --- Create a logic network ---
def random_individual(num_gates=8):
    individual = []
    available = ['A', 'B', 'nA', 'nB', 'C', 'nC']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate network ---
def evaluate_network(individual, a, b, c):
    signals = {'A': a, 'B': b, 'nA': 1 - a, 'nB': 1 - b, 'C': c, 'nC': 1 - c}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Evaluate fitness ---
def evaluate_fitness(individual):
    if len(individual) < 2:
        return 0  # Can't compute outputs
    score = 0
    for a, b, c in generate_inputs():
        signals = evaluate_network(individual, a, b, c)
        sum_out = signals[individual[-2]['name']]
        carry_out = signals[individual[-1]['name']]
        if sum_out == full_adder_sum(a, b, c):
            score += 1
        if carry_out == full_adder_carry(a, b, c):
            score += 1
    return score  # max = 16

# --- Mutate ---
def mutate(individual):
    mutant = []
    available = ['A', 'B', 'nA', 'nB', 'C', 'nC']

    for gate in individual:
        new_gate = {
            'name': gate['name'],
            'gate': gate['gate'],
            'inputs': gate['inputs'][:]  # deep copy inputs
        }

        if random.random() < 0.10:  # gate change
            new_gate['gate'] = random.choice(list(GATES.keys()))

        if random.random() < 0.20:  # input change
            idx = random.randint(0, 1)
            new_gate['inputs'][idx] = random.choice(available)

        mutant.append(new_gate)
        available.append(new_gate['name'])

    return mutant

# --- Shrink unused gates ---
def shrink(individual):
    if len(individual) < 2:
        return individual

    required = {individual[-1]['name'], individual[-2]['name']}
    changed = True

    while changed:
        changed = False
        for gate in individual:
            if gate['name'] in required:
                for inp in gate['inputs']:
                    for g in individual:
                        if g['name'] == inp and inp not in required:
                            required.add(inp)
                            changed = True

    return [g for g in individual if g['name'] in required]

# --- Evolution ---
def evolve(pop_size=100, generations=500, num_gates=8):
    population = [random_individual(num_gates) for _ in range(pop_size)]

    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        best_fit = evaluate_fitness(population[0])

        print(f"Generation {gen}: Best fitness = {best_fit}")

        if best_fit == 16:
            break

        next_gen = population[:4]  # elitism

        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = shrink(mutate(parent))
            next_gen.append(child)

        population = next_gen

    return population[0]


