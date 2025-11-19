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
def random_individual(num_gates=8):
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
def mutate(individual):
    mutant = []
    available = ['A', 'B', 'nA', 'nB']
    for i, gate in enumerate(individual):
        if random.random() < 0.2:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

# --- Evolution loop ---
def evolve(pop_size=100, generations=300, num_gates=8):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        best_fit = evaluate_fitness(population[0])
        if gen % 10 == 0 or best_fit == 8:
            print(f"Generation {gen}: Best fitness = {best_fit}")

        if best_fit == 8:
            break
        next_gen = population[:4]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = mutate(parent)
            next_gen.append(child)
        population = next_gen
    return population[0]


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

# --- Retry Evolution Until Perfect Network Found ---
best_network = None
fitness = 0
attempts = 0

while fitness != 8 and attempts < 10:
    candidate = evolve()
    fit = evaluate_fitness(candidate)
    print(f"\nAttempt {attempts + 1}: Fitness = {fit}")
    if fit == 8:
        best_network = candidate
        fitness = fit
        break
    attempts += 1
    if attempts == 10:
        print("❌ Failed to evolve a perfect MUX network after 10 attempts.")
        break

if best_network:
    print("\n🎯 Best network (fitness =", fitness, "/ 8):")
    print("Perfect Half Adder network found after", attempts, "attempts.")
    for gate in best_network:
        print(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

    print("\n🧪 Output Truth Table:")
    print(" A B | S | C | Target S | Target C")
    for a, b in generate_inputs():
        signals = evaluate_network(best_network, a, b)
        sum_out = signals[best_network[-2]['name']]
        carry_out = signals[best_network[-1]['name']]
        print(f" {a} {b} | {sum_out} | {carry_out} |     {half_adder_sum(a, b)}     |     {half_adder_carry(a, b)}")

    generate_structural_vhdl(best_network, filename="half_adder_custom.vhd")
