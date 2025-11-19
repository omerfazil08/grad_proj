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

# --- Inputs ---
INPUTS = ['A', 'B', 'nA', 'nB']

# --- Truth Table for Comparator (A > B) ---
def comparator_output(a, b):
    return 1 if a > b else 0

def generate_inputs():
    return [(a, b) for a in [0, 1] for b in [0, 1]]

# --- Individual Representation ---
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

def random_individual(num_gates=4):
    individual = []
    available = INPUTS.copy()
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate Network ---
def evaluate_network(individual, a, b):
    signals = {'A': a, 'B': b, 'nA': 1 - a, 'nB': 1 - b}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Fitness ---
def evaluate_fitness(individual):
    score = 0
    for a, b in generate_inputs():
        signals = evaluate_network(individual, a, b)
        out = signals[individual[-1]['name']]
        if out == comparator_output(a, b):
            score += 1
    return score  # max = 4

# --- Mutation ---
def mutate(individual):
    mutant = []
    available = INPUTS.copy()
    for i, gate in enumerate(individual):
        if random.random() < 0.3:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

# --- Evolution ---
def evolve(pop_size=50, generations=200, num_gates=4):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        best_fit = evaluate_fitness(population[0])
        if gen % 50 == 0 or best_fit == 4:
            print(f"Gen {gen}: Best fitness = {best_fit}")
        if best_fit == 4:
            return population[0]
        next_gen = population[:5]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = mutate(parent)
            next_gen.append(child)
        population = next_gen
    return population[0]

# 👇 Exported logic comparator
best_comparator = evolve()

def comparator_1bit_logic(a: int, b: int) -> int:
    signals = evaluate_network(best_comparator, a, b)
    return signals[best_comparator[-1]['name']]
# --- VHDL Generator ---
def generate_structural_vhdl(ind, filename="comparator_custom.vhd"):
    ports = ["A", "B"]
    inverted_ports = {"nA": "A", "nB": "B"}
    inv_signals = {inp for gate in ind for inp in gate["inputs"] if inp in inverted_ports}
    gate_names = [g["name"] for g in ind]

    lines = [
        "library IEEE;",
        "use IEEE.STD_LOGIC_1164.ALL;",
        "",
        "entity comparator_custom is",
        "  Port (",
        "    A   : in  STD_LOGIC;",
        "    B   : in  STD_LOGIC;",
        "    OUT : out STD_LOGIC",
        "  );",
        "end comparator_custom;",
        "",
        "architecture Structural of comparator_custom is"
    ]

    if inv_signals:
        lines.append(f"  signal {', '.join(sorted(inv_signals))} : STD_LOGIC;")
    lines.append(f"  signal {', '.join(gate_names)} : STD_LOGIC;")
    lines.append("begin")

    for inv in sorted(inv_signals):
        base = inverted_ports[inv]
        lines.append(f"  {inv} <= not {base};")

    for gate in ind:
        in1, in2 = gate["inputs"]
        op = gate["gate"].lower()
        # VHDL logical operators: use 'and', 'or', 'xor', etc.
        if op in ("and", "or", "xor"):
            expr = f"{in1} {op} {in2}"
        elif op == "nand":
            expr = f"not ({in1} and {in2})"
        elif op == "nor":
            expr = f"not ({in1} or {in2})"
        elif op == "xnor":
            expr = f"not ({in1} xor {in2})"
        else:
            raise ValueError(f"Unknown gate type: {op}")

        lines.append(f"  {gate['name']} <= {expr};")

    lines.append(f"  OUT <= {ind[-1]['name']};")
    lines.append("end Structural;")

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✅ VHDL written to {filename}")

# --- Run ---
best = evolve()
print("\n🎯 Best Comparator (A > B):")
for gate in best:
    print(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

print("\n🧪 Truth Table:")
print(" A B | OUT | Target")
for a, b in generate_inputs():
    signals = evaluate_network(best, a, b)
    out = signals[best[-1]['name']]
    print(f" {a} {b} |  {out}   |   {comparator_output(a, b)}")

generate_structural_vhdl(best)
