import random

# Define logic gates
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

# Define target functions for half adder
def half_adder_sum(a, b): return a ^ b
def half_adder_carry(a, b): return a & b

# Generate all 2-bit input combinations
def generate_inputs():
    return [(a, b) for a in [0, 1] for b in [0, 1]]

# Evaluate fitness of an individual
def evaluate_fitness(individual, target_func):
    score = 0
    for a, b in generate_inputs():
        inputs = {'A': a, 'B': b, 'nA': 1 - a, 'nB': 1 - b}
        try:
            out = GATES[individual['gate']](
                inputs[individual['inputs'][0]],
                inputs[individual['inputs'][1]]
            )
            if out == target_func(a, b):
                score += 1
        except:
            pass
    return score

# Create random individual
def random_individual():
    return {
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(['A', 'B', 'nA', 'nB'], 2)
    }

# Mutate individual
def mutate(individual):
    mutant = individual.copy()
    if random.random() < 0.5:
        mutant['gate'] = random.choice(list(GATES.keys()))
    else:
        mutant['inputs'] = random.sample(['A', 'B', 'nA', 'nB'], 2)
    return mutant

# Evolve best individual for a target function
def evolve(target_func, generations=50, population_size=20):
    population = [random_individual() for _ in range(population_size)]
    for generation in range(generations):
        population.sort(key=lambda ind: -evaluate_fitness(ind, target_func))
        if evaluate_fitness(population[0], target_func) == 4:
            break
        new_gen = population[:2]  # elitism
        while len(new_gen) < population_size:
            parent = random.choice(population[:5])
            child = mutate(parent)
            new_gen.append(child)
        population = new_gen
    return population[0]

# Run evolution
best_sum = evolve(half_adder_sum)
best_carry = evolve(half_adder_carry)

print("Best for SUM:", best_sum, "Fitness:", evaluate_fitness(best_sum, half_adder_sum))
print("Best for CARRY:", best_carry, "Fitness:", evaluate_fitness(best_carry, half_adder_carry))

best = max(population, key=evaluate)
print("\n🎯 Best individual:", best, "Fitness:", evaluate(best))

# --- VHDL Code Generator ---
def generate_structural_vhdl(ind):
    gate = ind["gate"]
    in1, in2 = ind["inputs"]

    vhdl = []
    vhdl.append("library IEEE;")
    vhdl.append("use IEEE.STD_LOGIC_1164.ALL;\n")
    vhdl.append("entity half_adder_custom is")
    vhdl.append("    Port ( A, B : in STD_LOGIC;")
    vhdl.append("           S, C : out STD_LOGIC );")
    vhdl.append("end half_adder_custom;\n")

    vhdl.append("architecture Structural of half_adder_custom is")
    signals = set()
    for i in [in1, in2]:
        if i.startswith("n"):
            signals.add(i)
    if signals:
        vhdl.append(f"    signal {', '.join(signals)} : STD_LOGIC;")
    vhdl.append("    signal sum_out : STD_LOGIC;")
    vhdl.append("    signal carry_out : STD_LOGIC;")
    vhdl.append("begin")

    # NOT operations
    for sig in signals:
        base = sig[1:]
        vhdl.append(f"    {sig} <= not {base};")

    # Gate logic
    if gate == "AND":
        vhdl.append(f"    sum_out <= {in1} and {in2};")
        vhdl.append(f"    carry_out <= {in1} and {in2};")
    elif gate == "OR":
        vhdl.append(f"    sum_out <= {in1} or {in2};")
        vhdl.append(f"    carry_out <= {in1} or {in2};")
    elif gate == "XOR":
        vhdl.append(f"    sum_out <= {in1} xor {in2};")
        vhdl.append(f"    carry_out <= {in1} xor {in2};")
    elif gate == "NAND":
        vhdl.append(f"    sum_out <= not ({in1} and {in2});")
        vhdl.append(f"    carry_out <= not ({in1} and {in2});")
    elif gate == "NOR":
        vhdl.append(f"    sum_out <= not ({in1} or {in2});")
        vhdl.append(f"    carry_out <= not ({in1} or {in2});")
    else:
        vhdl.append("    sum_out <= '0';")
        vhdl.append("    carry_out <= '0';")

    vhdl.append("    S <= sum_out;")
    vhdl.append("    C <= carry_out;")
    vhdl.append("end Structural;")

    return "\n".join(vhdl)

# --- Output VHDL ---
vhdl_code = generate_structural_vhdl(best)

with open("decoder_y_custom.vhd", "w") as f:
    f.write(vhdl_code)

print("\n✅ VHDL code generated: decoder_y_custom.vhd")
print("\n📄 VHDL Preview:\n")
print(vhdl_code)
