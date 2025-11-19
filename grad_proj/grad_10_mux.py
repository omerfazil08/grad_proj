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
def mux_out(a, b, sel): return a if sel == 0 else b

def generate_inputs():
    return [(a, b, sel) for a in [0, 1] for b in [0, 1] for sel in [0, 1]]

def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

def random_individual(num_gates=8):
    individual = []
    available = ['A', 'B', 'SEL', 'nA', 'nB', 'nSEL']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    individual[-2]['output'] = 'OUT1'
    individual[-1]['output'] = 'OUT2'
    return individual

def evaluate_network(individual, a, b, sel):
    signals = {'A': a, 'B': b, 'SEL': sel, 'nA': 1 - a, 'nB': 1 - b, 'nSEL': 1 - sel}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

def evaluate_fitness(individual):
    score = 0
    for a, b, sel in generate_inputs():
        signals = evaluate_network(individual, a, b, sel)
        out_gate = next(g for g in individual if g.get('output') == 'OUT2')
        out = signals[out_gate['name']]
        if out == mux_out(a, b, sel):
            score += 1
    return score

def mutate(individual):
    mutant = []
    available = ['A', 'B', 'SEL', 'nA', 'nB', 'nSEL']
    for i, gate in enumerate(individual):
        if random.random() < 0.2:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        if 'output' in gate:
            new_gate['output'] = gate['output']
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

def evolve(pop_size=100, generations=300, num_gates=8):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for _ in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        if evaluate_fitness(population[0]) == 8:
            return population[0]
        next_gen = population[:10]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = mutate(parent)
            next_gen.append(child)
        population = next_gen
    return population[0]

def generate_structural_vhdl(network, filename="mux_custom.vhd"):
    ports = ["A", "B", "SEL"]
    inverted_ports = {"nA": "A", "nB": "B", "nSEL": "SEL"}

    inv_signals = {inp for gate in network for inp in gate["inputs"] if inp in inverted_ports}
    gate_names = [g["name"] for g in network]

    lines = [
        "library IEEE;",
        "use IEEE.STD_LOGIC_1164.ALL;",
        "",
        "entity mux_custom is",
        "  Port (",
        "    A     : in  STD_LOGIC;",
        "    B     : in  STD_LOGIC;",
        "    SEL   : in  STD_LOGIC;",
        "    OUT   : out STD_LOGIC",
        "  );",
        "end mux_custom;",
        "",
        "architecture Structural of mux_custom is"
    ]

    if inv_signals:
        lines.append(f"  signal {', '.join(sorted(inv_signals))} : STD_LOGIC;")
    lines.append(f"  signal {', '.join(gate_names)} : STD_LOGIC;")
    lines.append("begin")

    for inv in sorted(inv_signals):
        base = inverted_ports[inv]
        lines.append(f"  {inv} <= not {base};")

    for gate in network:
        in1, in2 = gate["inputs"]
        op = gate["gate"].lower()
        expr = f"{in1} {op} {in2}" if op in ("and", "or", "xor") else f"not ({in1} {op[1:]} {in2})"
        lines.append(f"  {gate['name']} <= {expr};")

    out_gate = next(g for g in network if g.get('output') == 'OUT2')['name']
    lines.append(f"  OUT <= {out_gate};")
    lines.append("end Structural;")

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
    print("Perfect MUX network found after", attempts, "attempts.")
    for gate in best_network:
        print(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

    print("\n🧪 Output Truth Table:")
    print(" A B SEL | OUT | Target OUT")
    for a, b, sel in generate_inputs():
        signals = evaluate_network(best_network, a, b, sel)
        out_gate = next(g for g in best_network if g.get("output") == "OUT2")['name']
        out = signals[out_gate]
        print(f" {a} {b}  {sel}  |  {out}  |     {mux_out(a, b, sel)}")

    generate_structural_vhdl(best_network, filename="mux_custom.vhd")

