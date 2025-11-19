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

# --- Target Logic for MUX (2:1) ---
def mux_output(a, b, sel): return a if sel == 0 else b

# --- Generate 2-bit + selector combinations ---
def generate_inputs():
    return [(a, b, sel) for a in [0, 1] for b in [0, 1] for sel in [0, 1]]

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
    available = ['A', 'B', 'SEL', 'nA', 'nB', 'nSEL']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate the logic network ---
def evaluate_network(individual, a, b, sel):
    signals = {'A': a, 'B': b, 'SEL': sel, 'nA': 1 - a, 'nB': 1 - b, 'nSEL': 1 - sel}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Evaluate fitness ---
def evaluate_fitness(individual):
    score = 0
    for a, b, sel in generate_inputs():
        signals = evaluate_network(individual, a, b, sel)
        output = signals[individual[-1]['name']]
        if output == mux_output(a, b, sel):
            score += 1
    return score  # max = 8

# --- Mutate network ---
def mutate(individual):
    mutant = []
    available = ['A', 'B', 'SEL', 'nA', 'nB', 'nSEL']
    for i, gate in enumerate(individual):
        if random.random() < 0.3:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
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
            child = mutate(parent)
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

print("\n🧪 Output Truth Table:")
print(" A B SEL | OUT | Target OUT")
for a, b, sel in generate_inputs():
    signals = evaluate_network(best_network, a, b, sel)
    out = signals[best_network[-1]['name']]
    expected = mux_output(a, b, sel)
    print(f" {a} {b}  {sel}   |  {out}   |     {expected}")


def generate_mux_vhdl(network, filename="mux_custom.vhd"):
    """
    Generates structural VHDL for a 2:1 multiplexer from the evolved gate network.
    Assumes 'A', 'B', 'SEL', 'nA', 'nB', 'nSEL' may be used as inputs.
    Final output is connected to the gate with the highest fitness match.
    """
    # Define input/output port names
    ports = ["A", "B", "SEL"]
    inverted_ports = {"nA": "A", "nB": "B", "nSEL": "SEL"}
    inv_signals = set()
    
    for gate in network:
        for inp in gate["inputs"]:
            if inp in inverted_ports:
                inv_signals.add(inp)

    gate_names = [g["name"] for g in network]

    # --- VHDL Code Generation ---
    lines = []
    lines.append("library IEEE;")
    lines.append("use IEEE.STD_LOGIC_1164.ALL;")
    lines.append("")
    lines.append("entity mux_custom is")
    lines.append("  Port (")
    lines.append("    A    : in  STD_LOGIC;")
    lines.append("    B    : in  STD_LOGIC;")
    lines.append("    SEL  : in  STD_LOGIC;")
    lines.append("    Y    : out STD_LOGIC");
    lines.append("  );")
    lines.append("end mux_custom;")
    lines.append("")
    lines.append("architecture Structural of mux_custom is")

    if inv_signals:
        lines.append(f"  signal {', '.join(sorted(inv_signals))} : STD_LOGIC;")

    lines.append(f"  signal {', '.join(gate_names)} : STD_LOGIC;")
    lines.append("begin")

    # Add inversion logic if needed
    for inv in sorted(inv_signals):
        base = inverted_ports[inv]
        lines.append(f"  {inv} <= not {base};")

    for gate in network:
        in1, in2 = gate["inputs"]
        op = gate["gate"].lower()
        if op in ("and", "or", "xor"):
            expr = f"{in1} {op} {in2}"
        else:
            expr = f"not ({in1} {op[1:]} {in2})"
        lines.append(f"  {gate['name']} <= {expr};")

    # Assume last gate output is the best evolved output
    lines.append(f"  Y <= {network[-1]['name']};")
    lines.append("end Structural;")

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ VHDL code for MUX written to {filename}")
