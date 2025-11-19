import random

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
def half_adder_sum(a, b): return a ^ b
def half_adder_carry(a, b): return a & b

# --- Generate 2-bit combinations ---
def generate_inputs():
    return [(a, b) for a in [0, 1] for b in [0, 1]]

# --- Create a random gate ---
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

# --- Create a logic network (individual) ---
def random_individual(num_gates=4):
    individual = []
    available = ['A', 'B', 'nA', 'nB']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate the logic network ---
def evaluate_network(individual, a, b):
    signals = {'A': a, 'B': b, 'nA': 1 - a, 'nB': 1 - b}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Evaluate fitness ---
def evaluate_fitness(individual):
    score = 0
    for a, b in generate_inputs():
        signals = evaluate_network(individual, a, b)
        sum_out = signals[individual[-2]['name']]
        carry_out = signals[individual[-1]['name']]
        if sum_out == half_adder_sum(a, b):
            score += 1
        if carry_out == half_adder_carry(a, b):
            score += 1
    return score  # max = 8

# --- Mutate network ---
def advanced_mutate(individual, generation, max_generations):
    """
    Mutates an individual with adaptive and granular control.
    - Small tweaks instead of full replacements
    - Occasional crossover of gate inputs
    - Mutation rate decays over time
    """
    # Adaptive mutation rate
    mutation_rate = max(0.1, 0.5 * (1 - generation / max_generations))  # From 0.5 → 0.1
    
    mutant = []
    available = ['A', 'B', 'nA', 'nB']

    for i, gate in enumerate(individual):
        new_gate = gate.copy()

        # 1. With some chance, tweak the gate type
        if random.random() < mutation_rate:
            current_gate = gate['gate']
            other_gates = [g for g in GATES.keys() if g != current_gate]
            new_gate['gate'] = random.choice(other_gates)

        # 2. With some chance, tweak *one* input instead of both
        if random.random() < mutation_rate:
            inputs = new_gate['inputs']
            if random.random() < 0.5:
                inputs[0] = random.choice(available)
            else:
                inputs[1] = random.choice(available)
            new_gate['inputs'] = inputs

        # 3. Occasionally swap inputs (crossover-like effect)
        if random.random() < 0.05 and i > 0:
            swap_with = random.randint(0, i - 1)
            new_gate['inputs'] = individual[swap_with]['inputs'][:]

        available.append(new_gate['name'])
        mutant.append(new_gate)

    return mutant


# --- Evolution loop ---
def evolve(pop_size=30, generations=100, num_gates=4):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        best_fit = evaluate_fitness(population[0])
        if best_fit == 8:
            break
        next_gen = population[:4]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = advanced_mutate(parent, generation=gen, max_generations=generations)

            next_gen.append(child)
        population = next_gen
    return population[0]

# --- Run the evolution ---
best_network = evolve()
fitness = evaluate_fitness(best_network)

# --- Print best result ---
print("\n🎯 Best network (fitness =", fitness, "/ 8):")
for gate in best_network:
    print(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

def generate_structural_vhdl(network, filename="half_adder_custom.vhd"):
    """
    network: list of gates in order [g0, g1, g2, g3]
      each gate is a dict with keys:
        'name'   : e.g. 'g0'
        'gate'   : one of GATES.keys()
        'inputs' : two-element list of signal names
    """
    # Ports and known originals
    ports = ["A", "B"]
    inverted_ports = {"nA": "A", "nB": "B"}  # map inverted name → base port
    all_signal_names = set(ports)

    # Collect intermediate signals and which inversions we need
    inv_signals = set()
    for gate in network:
        for inp in gate["inputs"]:
            if inp in inverted_ports:
                inv_signals.add(inp)
        all_signal_names.add(gate["name"])

    # Begin building VHDL
    lines = []
    lines.append("library IEEE;")
    lines.append("use IEEE.STD_LOGIC_1164.ALL;")
    lines.append("")
    lines.append("entity half_adder_custom is")
    lines.append("  Port (")
    lines.append("    A     : in  STD_LOGIC;")
    lines.append("    B     : in  STD_LOGIC;")
    lines.append("    S     : out STD_LOGIC;")
    lines.append("    C     : out STD_LOGIC")
    lines.append("  );")
    lines.append("end half_adder_custom;")
    lines.append("")
    lines.append("architecture Structural of half_adder_custom is")

    # Declare inverted signals if needed
    if inv_signals:
        inv_list = ", ".join(sorted(inv_signals))
        lines.append(f"  signal {inv_list} : STD_LOGIC;")

    # Declare each gate’s output signal
    gate_names = [g["name"] for g in network]
    lines.append(f"  signal {', '.join(gate_names)} : STD_LOGIC;")
    lines.append("begin")

    # Generate NOT assignments for any inverted ports
    for inv in sorted(inv_signals):
        base = inverted_ports[inv]
        lines.append(f"  {inv} <= not {base};")

    # Generate each gate’s logic
    for gate in network:
        in1, in2 = gate["inputs"]
        op = gate["gate"].lower()
        # VHDL logical operators: use 'and', 'or', 'xor', etc.
        if op in ("and", "or", "xor"):
            expr = f"{in1} {op} {in2}"
        else:
            # NAND, NOR, XNOR become `not ( … )`
            expr = f"not ({in1} {op[1:]} {in2})"
        lines.append(f"  {gate['name']} <= {expr};")

    # Last two gates map to S and C
    sum_gate   = network[-2]["name"]
    carry_gate = network[-1]["name"]
    lines.append(f"  S <= {sum_gate};")
    lines.append(f"  C <= {carry_gate};")

    lines.append("end Structural;")

    # Write file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ VHDL code written to {filename}")
# --- After printing the best network ---
generate_structural_vhdl(best_network, filename="half_adder_custom.vhd")
