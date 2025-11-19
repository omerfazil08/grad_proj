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

# --- User Target Input ---
def get_user_target():
    num_inputs = int(input("Enter number of inputs (2 or 3): "))
    num_outputs = int(input("Enter number of outputs (1 or 2): "))

    # Construct all input combinations
    if num_inputs == 2:
        inputs = [(a, b, 0) for a in [0,1] for b in [0,1]]  # dummy C=0
    else:
        inputs = [(a, b, c) for a in [0,1] for b in [0,1] for c in [0,1]]

    print("\nInput rows order will be:")
    for row in inputs:
        print(row)

    target_outputs = []
    for o in range(num_outputs):
        print(f"\nEnter output truth table #{o+1} values, space-separated:")
        print("Expected number of values:", len(inputs))
        values = list(map(int, input("→ ").split()))
        if len(values) != len(inputs):
            raise ValueError("❌ Incorrect number of truth table entries.")
        target_outputs.append(values)

    return num_inputs, num_outputs, inputs, target_outputs


# --- Create Random Gate ---
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

# --- Create Individual ---
def random_individual(num_gates=8):
    individual = []
    available = ['A', 'B', 'nA', 'nB', 'C', 'nC']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate Network ---
def evaluate_network(individual, a, b, c):
    signals = {'A': a, 'B': b, 'nA': 1-a, 'nB': 1-b, 'C': c, 'nC': 1-c}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Fitness Function (Generic) ---
def evaluate_fitness(individual, num_inputs, num_outputs, inputs, targets):
    score = 0
    for idx, (a, b, c) in enumerate(inputs):
        signals = evaluate_network(individual, a, b, c)
        # Last gates produce outputs
        for o in range(num_outputs):
            gate_name = individual[-num_outputs + o]['name']
            if signals[gate_name] == targets[o][idx]:
                score += 1
    return score

# --- Mutation ---
def mutate(individual):
    mutant = []
    available = ['A', 'B', 'nA', 'nB', 'C', 'nC']
    for i, gate in enumerate(individual):
        if random.random() < 0.2:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

# --- Evolution ---
def evolve(num_inputs, num_outputs, inputs, targets, pop_size=100, generations=500, num_gates=8):
    population = [random_individual(num_gates) for _ in range(pop_size)]

    max_score = len(inputs) * num_outputs  # perfect score

    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind, num_inputs, num_outputs, inputs, targets))
        best = population[0]
        best_score = evaluate_fitness(best, num_inputs, num_outputs, inputs, targets)

        if gen % 50 == 0 or best_score == max_score:
            print(f"Generation {gen}: Best fitness = {best_score}/{max_score}")

        if best_score == max_score:
            break

        next_gen = population[:4]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            next_gen.append(mutate(parent))
        population = next_gen

    return population[0], best_score, max_score


# --- MAIN RUN ---
num_inputs, num_outputs, inputs, targets = get_user_target()

best, score, max_score = evolve(num_inputs, num_outputs, inputs, targets)

print("\n✅ Best Network Found:", score, "/", max_score)
print("GATE LIST:")
for g in best:
    print(f"{g['name']}: {g['gate']}({g['inputs'][0]}, {g['inputs'][1]})")

print("\nTruth Table Check:")
for idx, (a, b, c) in enumerate(inputs):
    out = evaluate_network(best, a, b, c)
    row = [out[best[-num_outputs + o]['name']] for o in range(num_outputs)]
    print(f"{(a,b,c)} → {row}   (target {[targets[o][idx] for o in range(num_outputs)]})")

# ---------- NEW: SIMPLIFICATION ----------
print("\n🧠 Simplified Output Logic:")
for o in range(num_outputs):
    output_gate = best[-num_outputs + o]["name"]
    simplified = simplify_single_output(best, output_gate)
    print(f"Output {o+1} ({output_gate}):\n  {simplified}")
