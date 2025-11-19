# grad_38_evolve_custom_logic.py
import random
import copy
from grad_34_simp2 import simplify_single_output

# --- Basic gates (bitwise on ints 0/1) ---
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
# Utilities: inputs / names
# -------------------------
def build_input_names(n):
    """Return list of input names: A, B, C, D, ... (up to reasonable n)."""
    names = []
    base = ord('A')
    for i in range(n):
        # After 'Z' this would break — but typical n is small (<= 10)
        names.append(chr(base + i))
    return names

def neg_name(name):
    return f"n{name}"

# -------------------------
# Individual representation
# -------------------------
# individual = {
#   'gates': [ {'name':'g0','gate':'XOR','inputs':['A','B']}, ... ],
#   'outputs': ['g6', 'A']   # list of names (can be gate names or input names)
# }

def random_gate(index, available_inputs):
    return {
        'name': f"g{index}",
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

def fix_and_repair_gates(gates, input_names):
    """
    Rename gates to g0..gN-1 in order and ensure every gate's inputs refer to either:
    - input_names or
    - previously created gates in this order.
    If an input refers to a non-existent name, replace it with a random available input.
    Return the repaired gate list and a mapping old->new names.
    """
    new_gates = []
    available = input_names[:] + [neg_name(n) for n in input_names]
    name_map = {}  # old_name -> new_name

    for i, g in enumerate(gates):
        new_name = f"g{i}"
        name_map[g['name']] = new_name

        # Determine inputs: if referenced name already mapped -> use mapped name.
        repaired_inputs = []
        for inp in g['inputs']:
            if inp in name_map:
                repaired_inputs.append(name_map[inp])
            elif inp in available:
                repaired_inputs.append(inp)
            else:
                # pick a random available input
                repaired_inputs.append(random.choice(available))
        new_gates.append({'name': new_name, 'gate': g['gate'], 'inputs': repaired_inputs})
        available.append(new_name)

    return new_gates, name_map

def random_individual(num_inputs, num_outputs, num_gates=8):
    input_names = build_input_names(num_inputs)
    available = input_names[:] + [neg_name(n) for n in input_names]
    gates = []
    for i in range(num_gates):
        g = random_gate(i, available)
        gates.append(g)
        available.append(g['name'])

    # pick outputs from all available signals (inputs + gates)
    all_signals = input_names + [neg_name(n) for n in input_names] + [g['name'] for g in gates]
    outputs = random.sample(all_signals, num_outputs)
    return {'gates': gates, 'outputs': outputs}

# -------------------------
# Evaluation
# -------------------------
def evaluate_network(individual, input_values):
    """
    input_values: dict mapping input name -> 0/1, e.g. {'A':0, 'B':1, 'C':0}
    Returns signals dict with gate outputs and inputs included.
    """
    signals = {}
    # initialize primary inputs and negations
    for k, v in input_values.items():
        signals[k] = v
        signals[neg_name(k)] = 1 - v

    # evaluate gates in list order
    for gate in individual['gates']:
        in1 = signals.get(gate['inputs'][0])
        in2 = signals.get(gate['inputs'][1])
        # in rare cases of missing inputs (shouldn't happen if repaired), fallback
        if in1 is None:
            in1 = 0
        if in2 is None:
            in2 = 0
        signals[gate['name']] = GATES[gate['gate']](in1, in2)

    return signals

# -------------------------
# Fitness
# -------------------------
def evaluate_fitness(individual, input_names, inputs_list, targets, gate_penalty=0.0):
    """
    - inputs_list: list of tuples representing all input vectors in the order used for targets.
      e.g., [(0,0),(0,1),(1,0),(1,1)] for 2 inputs.
    - targets: list of lists: targets[o][i] is expected bit for output o and input vector i.
    - gate_penalty: subtract penalty * number_of_gates  (to prefer smaller circuits)
    """
    score = 0
    num_outputs = len(individual['outputs'])
    for idx, vec in enumerate(inputs_list):
        iv = {name: bit for name, bit in zip(input_names, vec)}
        signals = evaluate_network(individual, iv)
        for o_index, out_name in enumerate(individual['outputs']):
            bit = signals.get(out_name, 0)
            if bit == targets[o_index][idx]:
                score += 1
    # apply penalty
    score -= gate_penalty * len(individual['gates'])
    return score

# -------------------------
# Mutation (finer-grained)
# -------------------------
def mutate(individual, input_names, base_mutation_rate):
    """
    Mutate gates and outputs with several smaller mutation operators.
    base_mutation_rate is the current mutation probability used as a guide.
    """
    ind = copy.deepcopy(individual)
    available_signals = input_names[:] + [neg_name(n) for n in input_names] + [g['name'] for g in ind['gates']]

    for i, gate in enumerate(ind['gates']):
        if random.random() < base_mutation_rate:
            op = random.random()
            if op < 0.25:
                # change gate type
                gate['gate'] = random.choice(list(GATES.keys()))
            elif op < 0.6:
                # change one input
                idx = random.choice([0, 1])
                gate['inputs'][idx] = random.choice(available_signals[:max(1, i + len(input_names))])
            elif op < 0.85:
                # swap inputs
                gate['inputs'][0], gate['inputs'][1] = gate['inputs'][1], gate['inputs'][0]
            else:
                # replace whole gate
                gate = random_gate(i, available_signals[:max(1, i + len(input_names))])
                ind['gates'][i] = gate

    # mutate outputs: with small chance, replace one output with another valid signal
    for idx in range(len(ind['outputs'])):
        if random.random() < (base_mutation_rate * 0.5):
            all_signals = input_names + [neg_name(n) for n in input_names] + [g['name'] for g in ind['gates']]
            ind['outputs'][idx] = random.choice(all_signals)

    # repair to ensure ordering and valid references
    ind['gates'], name_map = fix_and_repair_gates(ind['gates'], input_names)
    # remap outputs if they pointed to old gate names
    new_outputs = []
    for o in ind['outputs']:
        if o in name_map:
            new_outputs.append(name_map[o])
        else:
            new_outputs.append(o if o in input_names or o.startswith('n') else random.choice(input_names))
    ind['outputs'] = new_outputs

    return ind

# -------------------------
# Crossover (one-point + repair)
# -------------------------
def crossover(parent1, parent2, input_names):
    """
    One-point crossover on gate lists. After combining, repair naming and invalid references.
    Return a new child individual.
    """
    g1 = parent1['gates']
    g2 = parent2['gates']
    if len(g1) < 2 or len(g2) < 2:
        child_gates = copy.deepcopy(g1 if len(g1) >= len(g2) else g2)
    else:
        cut1 = random.randint(1, len(g1)-1)
        cut2 = random.randint(1, len(g2)-1)
        # simpler: pick one cut and splice
        cut = random.choice([cut1, cut2])
        child_gates = copy.deepcopy(g1[:cut] + g2[cut:])

    # Repair and rename gates to g0..gN-1 and repair inputs
    child_gates, name_map = fix_and_repair_gates(child_gates, input_names)

    # build outputs: prefer mapping from parent outputs if possible, else pick random valid signals
    child_outputs = []
    candidate_signals = input_names + [neg_name(n) for n in input_names] + [g['name'] for g in child_gates]
    for o in parent1['outputs'] + parent2['outputs']:
        if o in name_map:
            child_outputs.append(name_map[o])
        elif o in candidate_signals and o not in child_outputs:
            child_outputs.append(o)
        # stop if enough outputs
        if len(child_outputs) >= max(len(parent1['outputs']), len(parent2['outputs'])):
            break

    # if we still don't have outputs, fill randomly
    desired_outputs = max(len(parent1['outputs']), len(parent2['outputs']))
    while len(child_outputs) < desired_outputs:
        cand = random.choice(candidate_signals)
        if cand not in child_outputs:
            child_outputs.append(cand)

    return {'gates': child_gates, 'outputs': child_outputs[:desired_outputs]}

# -------------------------
# Selection: tournament
# -------------------------
def tournament_select(population, fitness_map, k=3):
    """Pick k individuals at random and return the best one (by fitness_map)."""
    candidates = random.sample(population, k)
    best = max(candidates, key=lambda ind: fitness_map[id(ind)])
    return best

# -------------------------
# Optional seeding helpers
# -------------------------
def seed_half_adder(input_names=None):
    """
    Returns an individual (gates + outputs) implementing a half-adder for A,B:
    SUM = A xor B -> g0
    CARRY = A and B -> g1
    Works if input_names contains at least 'A' and 'B'.
    """
    if input_names is None:
        input_names = ['A', 'B']
    A = input_names[0]
    B = input_names[1]
    g0 = {'name': 'g0', 'gate': 'XOR', 'inputs': [A, B]}
    g1 = {'name': 'g1', 'gate': 'AND', 'inputs': [A, B]}
    # outputs are g0 (sum), g1 (carry)
    return {'gates': [g0, g1], 'outputs': ['g0', 'g1']}

# -------------------------
# Main GA evolve function
# -------------------------
def evolve(num_inputs, num_outputs, inputs_list, targets,
           pop_size=120, generations=500, num_gates=8,
           base_mutation=0.25, decay=0.995, elitism=4, gate_penalty=0.0,
           seed_fraction=0.0):
    """
    - inputs_list: list of tuples (bit vectors) for all combinations (order must match targets).
    - targets: list of lists, targets[o][i] expected bit for output o and input index i
    """
    input_names = build_input_names(num_inputs)

    # initialize population
    population = []
    n_seed = int(pop_size * seed_fraction)
    for _ in range(n_seed):
        # currently only seeding half-adders as an example; in future add more modules
        s = seed_half_adder(input_names[:2])
        # if required, repair to desired num_gates by expanding with random gates
        # expand to num_gates
        while len(s['gates']) < num_gates:
            idx = len(s['gates'])
            available = input_names + [neg_name(n) for n in input_names] + [g['name'] for g in s['gates']]
            s['gates'].append(random_gate(idx, available))
        s['gates'], _ = fix_and_repair_gates(s['gates'], input_names)
        # ensure outputs length
        if len(s['outputs']) < num_outputs:
            cand = input_names + [g['name'] for g in s['gates']]
            while len(s['outputs']) < num_outputs:
                s['outputs'].append(random.choice(cand))
        s['outputs'] = s['outputs'][:num_outputs]
        population.append(s)

    # fill rest randomly
    while len(population) < pop_size:
        population.append(random_individual(num_inputs, num_outputs, num_gates))

    max_score = len(inputs_list) * num_outputs

    # main loop
    best_overall = None
    best_score = float('-inf')

    for gen in range(generations):
        # adaptive mutation rate
        mutation_rate = base_mutation * (decay ** gen)

        # evaluate fitness for all individuals once (cache)
        fitness_map = {}
        for ind in population:
            f = evaluate_fitness(ind, input_names, inputs_list, targets, gate_penalty)
            fitness_map[id(ind)] = f

        # sort population by fitness descending
        population.sort(key=lambda ind: -fitness_map[id(ind)])
        current_best = population[0]
        current_best_score = fitness_map[id(current_best)]

        if current_best_score > best_score:
            best_score = current_best_score
            best_overall = copy.deepcopy(current_best)

        if gen % 25 == 0 or current_best_score == max_score:
            print(f"Gen {gen:4d}: best = {current_best_score}/{max_score} (mutation={mutation_rate:.4f})")

        if current_best_score == max_score:
            print("Perfect solution found.")
            best_overall = current_best
            break

        # Elitism: keep top 'elitism'
        next_gen = population[:elitism]

        # Fill the rest: use tournament selection + crossover + mutation
        while len(next_gen) < pop_size:
            parent1 = tournament_select(population, fitness_map, k=3)
            parent2 = tournament_select(population, fitness_map, k=3)
            # avoid identical parents
            if id(parent1) == id(parent2):
                child = mutate(parent1, input_names, mutation_rate)
            else:
                child = crossover(parent1, parent2, input_names)
                # mutate child
                child = mutate(child, input_names, mutation_rate)
            next_gen.append(child)

        population = next_gen

    return best_overall, int(best_score), int(max_score), input_names

# -------------------------
# Helpers for user interaction
# -------------------------
def all_input_combinations(num_inputs):
    """Return list of tuples with all combinations for num_inputs bits (lexicographic)."""
    combos = []
    for i in range(2 ** num_inputs):
        bits = tuple((i >> j) & 1 for j in reversed(range(num_inputs)))
        combos.append(bits)
    return combos

def get_user_target_interactive():
    num_inputs = int(input("Enter number of inputs (>=2): "))
    num_outputs = int(input("Enter number of outputs (>=1): "))

    input_names = build_input_names(num_inputs)
    inputs_list = all_input_combinations(num_inputs)
    print("Input rows order will be:")
    for row in inputs_list:
        print(row)

    target_outputs = []
    for o in range(num_outputs):
        print(f"\nEnter output truth table #{o+1} values, space-separated:")
        print("Expected number of values:", len(inputs_list))
        values = list(map(int, input("→ ").split()))
        if len(values) != len(inputs_list):
            raise ValueError("Incorrect number of truth table entries.")
        target_outputs.append(values)

    return num_inputs, num_outputs, inputs_list, target_outputs

# -------------------------
# If run as script
# -------------------------
if __name__ == "__main__":
    print("=== grad_38: Genetic evolution of logic networks ===")
    # get interactive target
    num_inputs, num_outputs, inputs_list, targets = get_user_target_interactive()

    # Example: change parameters here if desired
    best, score, max_score, input_names = evolve(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        inputs_list=inputs_list,
        targets=targets,
        pop_size=160,
        generations=800,
        num_gates=10,
        base_mutation=0.30,
        decay=0.997,
        elitism=6,
        gate_penalty=0.0,
        seed_fraction=0.08  # small seeding fraction (optional)
    )

    print("\n✅ Best network (score):", score, "/", max_score)
    print("GATE LIST:")
    for g in best['gates']:
        print(f"{g['name']}: {g['gate']}({g['inputs'][0]}, {g['inputs'][1]})")
    print("OUTPUTS:", best['outputs'])

    print("\nTruth Table Check:")
    for idx, vec in enumerate(inputs_list):
        iv = {name: bit for name, bit in zip(input_names, vec)}
        out_signals = evaluate_network(best, iv)
        row = [out_signals.get(o, 0) for o in best['outputs']]
        print(f"{vec} → {row}   (target {[targets[o][idx] for o in range(len(best['outputs']))]})")

    print("\n🧠 Simplified Output Logic:")
    for o_index, out_name in enumerate(best['outputs']):
        simplified = simplify_single_output(best['gates'], out_name)
        print(f"Output {o_index+1} ({out_name}):\n  {simplified}")
