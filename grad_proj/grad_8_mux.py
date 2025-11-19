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

# --- Target Logic for 2:1 MUX ---
def mux_output(a, b, sel):
    return a if sel == 0 else b

# --- Generate input combinations ---
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
    # Tag last gate as MUX output
    individual[-1]['output'] = 'OUT'
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
        out_gate = next(g for g in individual if g.get('output') == 'OUT')
        out_val = signals[out_gate['name']]
        if out_val == mux_output(a, b, sel):
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
        # Preserve output tag
        if 'output' in gate:
            new_gate['output'] = gate['output']
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

# --- Evolution loop ---
def evolve(pop_size=30, generations=100, num_gates=4):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        if evaluate_fitness(population[0]) == len(generate_inputs()):
            break
        next_gen = population[:4]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = mutate(parent)
            next_gen.append(child)
        population = next_gen
    return population[0]

# --- Run evolution ---
best_network = evolve()
fitness = evaluate_fitness(best_network)

# --- Print best network ---
print(f"\n🎯 Best network (fitness = {fitness} / 8):")
for gate in best_network:
    tag = '*' if gate.get('output') == 'OUT' else ' '
    print(f"{tag}{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

# --- Output truth table ---
print("\n🧪 Output Truth Table:")
print(" A B SEL | OUT | Target")
for a, b, sel in generate_inputs():
    signals = evaluate_network(best_network, a, b, sel)
    out_gate = next(g for g in best_network if g.get('output') == 'OUT')
    out_val = signals[out_gate['name']]
    print(f" {a} {b}  {sel}   |  {out_val}   |    {mux_output(a, b, sel)}")

# --- VHDL Generator ---
def generate_structural_vhdl(network, filename="mux_custom.vhd"):
    # Ports and inversions
    ports = ['A', 'B', 'SEL']
    inv_ports = {'nA':'A', 'nB':'B', 'nSEL':'SEL'}
    # Collect inversion and gate signals
    inv_signals = {inp for g in network for inp in g['inputs'] if inp in inv_ports}
    gate_names = [g['name'] for g in network]

    lines = [
        "library IEEE;",
        "use IEEE.STD_LOGIC_1164.ALL;",
        "",
        "entity mux_custom is",
        "  Port ( A   : in  STD_LOGIC;",
        "         B   : in  STD_LOGIC;",
        "         SEL : in  STD_LOGIC;",
        "         OUT : out STD_LOGIC );",
        "end mux_custom;",
        "",
        "architecture Structural of mux_custom is"
    ]
    if inv_signals:
        lines.append(f"  signal {', '.join(sorted(inv_signals))} : STD_LOGIC;")
    lines.append(f"  signal {', '.join(gate_names)} : STD_LOGIC;")
    lines.append("begin")
    # NOT assignments
    for inv in sorted(inv_signals):
        base = inv_ports[inv]
        lines.append(f"  {inv} <= not {base};")
    # Gate logic
    for g in network:
        in1, in2 = g['inputs']
        op = g['gate'].lower()
        if op in ('and','or','xor'):
            expr = f"{in1} {op} {in2}"
        else:
            expr = f"not ({in1} {op[1:]} {in2})"
        lines.append(f"  {g['name']} <= {expr};")
    # Map output
    out_name = next(g['name'] for g in network if g.get('output')=='OUT')
    lines.append(f"  OUT <= {out_name};")
    lines.append("end Structural;")

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✅ VHDL code written to {filename}")

# --- Generate VHDL ---
generate_structural_vhdl(best_network)
