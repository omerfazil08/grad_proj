# Re-running the code after kernel reset

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

# --- D-Latch Target Logic ---
def d_latch_logic(D, EN, prev_Q):
    if EN == 1:
        return D, 1 - D
    else:
        return prev_Q, 1 - prev_Q

# --- Generate Input Combinations ---
def generate_inputs():
    return [(d, en, q) for d in [0, 1] for en in [0, 1] for q in [0, 1]]

# --- Create Random Gate ---
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, 2)
    }

# --- Create Logic Network ---
def random_individual(num_gates=6):
    individual = []
    available = ['D', 'EN', 'nD', 'nEN', 'Q', 'nQ']
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

# --- Evaluate Logic Network ---
def evaluate_network(individual, D, EN, Q):
    signals = {'D': D, 'EN': EN, 'nD': 1 - D, 'nEN': 1 - EN, 'Q': Q, 'nQ': 1 - Q}
    for gate in individual:
        in1 = signals[gate['inputs'][0]]
        in2 = signals[gate['inputs'][1]]
        signals[gate['name']] = GATES[gate['gate']](in1, in2)
    return signals

# --- Evaluate Fitness ---
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
    return score  # max = 12

# --- Mutate Network ---
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

# --- Evolution Loop ---
def evolve(pop_size=100, generations=500, num_gates=6):
    population = [random_individual(num_gates) for _ in range(pop_size)]
    for gen in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind))
        best_fit = evaluate_fitness(population[0])
        if gen % 50 == 0 or best_fit == 12:
            print(f"Generation {gen}: Best fitness = {best_fit}")
        if best_fit == 12:
            break
        next_gen = population[:4]
        while len(next_gen) < pop_size:
            parent = random.choice(population[:10])
            child = mutate(parent)
            next_gen.append(child)
        population = next_gen
    return population[0]

# Run the evolution and show result
best_network = None
fitness = 0
attempts = 0

while fitness != 12 and attempts < 10:
    candidate = evolve()
    fit = evaluate_fitness(candidate)
    if fit == 12:
        best_network = candidate
        fitness = fit
        break
    attempts += 1
    if attempts == 10:
       print("❌ Failed to evolve a perfect Full Adder network after 10 attempts.")
       break

def generate_structural_vhdl(network, filename="dlatch_custom.vhd"):
    """
    network: list of gates in order [g0, g1, g2, g3]
      each gate is a dict with keys:
        'name'   : e.g. 'g0'
        'gate'   : one of GATES.keys()
        'inputs' : two-element list of signal names
    """
    # Ports and known originals
    ports = ["D", "En", "Q"]
    inverted_ports = {"nD": "D", "nEn": "En", "nQ": "Q"}  # map inverted name → base port
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
    lines.append("entity dlatch_custom is")
    lines.append("  Port (")
    lines.append("    D     : in  STD_LOGIC;")
    lines.append("    En    : in  STD_LOGIC;")
    lines.append("    Q     : out STD_LOGIC")
    lines.append("  );")
    lines.append("end dlatch_custom;")
    lines.append("")
    lines.append("architecture Structural of dlatch_custom is")

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
        elif op == "nand":
            expr = f"not ({in1} and {in2})"
        elif op == "nor":
            expr = f"not ({in1} or {in2})"
        elif op == "xnor":
            expr = f"not ({in1} xor {in2})"
        else:
            raise ValueError(f"Unknown gate type: {op}")

        lines.append(f"  {gate['name']} <= {expr};")

    # Last two gates map to Sum and Carry
    Q_o   = network[-1]["name"]
    lines.append(f"  Q <= {Q_o};")

    lines.append("end Structural;")

    # Write file
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ VHDL code written to {filename}")

# Display results if successful
summary = []
if best_network:
    summary.append(f"Best network (fitness = {fitness} / 12):")
    for gate in best_network:
        summary.append(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")
else:
    summary.append("Failed to evolve a perfect D-Latch network after 10 attempts.")

if best_network:
    print("\n🎯 Best network (fitness =", fitness, "/ 12):")
    print("Perfect d-latch network found after", attempts, "attempts.")
    for gate in best_network:
        print(f"{gate['name']}: {gate['gate']}({gate['inputs'][0]}, {gate['inputs'][1]})")

    print("\n🧪 Output Truth Table:")
    print(" D Clk  | Q | Target Q")
    for d, en, q in generate_inputs():
        signals = evaluate_network(best_network, d, en, q)
        q = signals[best_network[-1]['name']]
        print(f" {d} {en} |{q} ")

    generate_structural_vhdl(best_network, filename="d-latch_custom.vhd")