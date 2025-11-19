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
def d_latch_logic(D, EN, prev_Q):
    if EN == 1:
        return D, 1 - D
    else:
        return prev_Q, 1 - prev_Q

# --- Inputs ---
def generate_inputs():
    return [(d, en, q) for d in [0, 1] for en in [0, 1] for q in [0, 1]]

# --- Random Gate ---
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

# --- Create Network ---
def random_individual(num_gates=6):
    individual = []
    available = ['D', 'EN', 'nD', 'nEN', 'Q', 'nQ']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate Network ---
def evaluate_network(individual, D, EN, Q):
    signals = {'D': D, 'EN': EN, 'nD': 1 - D, 'nEN': 1 - EN, 'Q': Q, 'nQ': 1 - Q}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Fitness ---
def evaluate_fitness(individual):
    score = 0
    for D, EN, Q in generate_inputs():
        signals = evaluate_network(individual, D, EN, Q)
        out_Q = signals[individual[-2]['name']]
        out_Qn = signals[individual[-1]['name']]
        target_Q, target_Qn = d_latch_logic(D, EN, Q)
        if out_Q == target_Q:
            score += 1
        if out_Qn == target_Qn:
            score += 1
    return score  # max = 16

# --- Mutation ---
def mutate(individual):
    mutant = []
    available = ['D', 'EN', 'nD', 'nEN', 'Q', 'nQ']
    for i, gate in enumerate(individual):
        if random.random() < 0.2:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

# --- Evolution ---
def evolve(pop_size=100, generations=500, num_gates=6):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        best_fit = evaluate_fitness(population[0])
        if gen % 50 == 0 or best_fit == 16:
            print(f"Generation {gen}: Best fitness = {best_fit}")
        if best_fit == 16:
            return population[0]
        next_gen = population[:4]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = mutate(parent)
            next_gen.append(child)
        population = next_gen
    return population[0]

# --- VHDL Generator ---
def generate_structural_vhdl(network, filename="dlatch_custom.vhd"):
    ports = ["D", "En", "Q", "Qn"]
    inverted_ports = {"nD": "D", "nEN": "En", "nQ": "Q"}
    inv_signals = {inp for gate in network for inp in gate["inputs"] if inp in inverted_ports}
    gate_names = [g["name"] for g in network]

    lines = [
        "library IEEE;",
        "use IEEE.STD_LOGIC_1164.ALL;",
        "",
        "entity dlatch_custom is",
        "  Port (",
        "    D   : in  STD_LOGIC;",
        "    En  : in  STD_LOGIC;",
        "    Q   : out STD_LOGIC;",
        "    Qn  : out STD_LOGIC",
        "  );",
        "end dlatch_custom;",
        "",
        "architecture Structural of dlatch_custom is"
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

    lines.append(f"  Q <= {network[-2]['name']};")
    lines.append(f"  Qn <= {network[-1]['name']};")
    lines.append("end Structural;")

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ VHDL code written to {filename}")

# --- Main Execution ---
best_network = None
fitness = 0
attempts = 0

while fitness != 16 and attempts < 10:
    candidate = evolve()
    fit = evaluate_fitness(candidate)
    print(f"\nAttempt {attempts + 1}: Fitness = {fit}")
    if fit == 16:
        best_network = candidate
        fitness = fit
        break
    attempts += 1
    if attempts == 10:
        print("❌ Failed to evolve a perfect D-Latch network after 10 attempts.")
        break

# --- Print Results ---
if best_network:
    print("\n🎯 Best network (fitness =", fitness, "/ 16):")
    for gate in best_network:
        print(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

    print("\n🧪 Output Truth Table:")
    print(" D EN PrevQ | Q | TargetQ")
    for d, en, q in generate_inputs():
        signals = evaluate_network(best_network, d, en, q)
        out_Q = signals[best_network[-2]['name']]
        target_Q, _ = d_latch_logic(d, en, q)
        print(f" {d}  {en}   {q}    | {out_Q} |    {target_Q}")

    generate_structural_vhdl(best_network, filename="dlatch_custom.vhd")



