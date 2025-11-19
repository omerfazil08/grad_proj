import random

# --- GA Parameters ---
POP_SIZE = 30
GENERATIONS = 200
MUTATION_RATE = 0.2

# --- Inputs and Gates ---
INPUTS = ["A", "B", "C", "nA", "nB", "nC"]
GATES = ["AND", "OR", "XOR", "NAND", "NOR"]

# --- Ask user for desired 8-bit output ---
print("Enter target truth table for A,B,C (8 bits from 000 to 111):")
print("Example (Y3 = not A and B and C): 00100000")
user_bits = input("Target output (length 8): ").strip()
while len(user_bits) != 8 or any(c not in "01" for c in user_bits):
    user_bits = input("Invalid input. Please enter exactly 8 bits (0 or 1): ").strip()

target_outputs = [int(b) for b in user_bits]

def target_func(A, B, C):
    index = (A << 2) | (B << 1) | C  # 3-bit to index (0–7)
    return target_outputs[index]

# --- Individual Representation ---
def random_individual():
    return {
        "gate": random.choice(GATES),
        "inputs": random.sample(INPUTS, 3)
    }

# --- Evaluate fitness ---
def evaluate(individual):
    score = 0
    for A in [0, 1]:
        for B in [0, 1]:
            for C in [0, 1]:
                vals = {
                    "A": A,
                    "B": B,
                    "C": C,
                    "nA": 1 - A,
                    "nB": 1 - B,
                    "nC": 1 - C
                }
                in1, in2, in3 = [vals[i] for i in individual["inputs"]]
                gate = individual["gate"]
                if gate == "AND":
                    out = in1 & in2 & in3
                elif gate == "OR":
                    out = in1 | in2 | in3
                elif gate == "XOR":
                    out = in1 ^ in2 ^ in3
                elif gate == "NAND":
                    out = 1 - (in1 & in2 & in3)
                elif gate == "NOR":
                    out = 1 - (in1 | in2 | in3)
                else:
                    out = 0

                if out == target_func(A, B, C):
                    score += 1
    return score  # max = 8

# --- Mutation ---
def mutate(ind):
    if random.random() < MUTATION_RATE:
        ind["gate"] = random.choice(GATES)
    if random.random() < MUTATION_RATE:
        ind["inputs"] = random.sample(INPUTS, 3)
    return ind

# --- GA Loop ---
population = [random_individual() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scored = [(ind, evaluate(ind)) for ind in population]
    scored.sort(key=lambda x: -x[1])
    population = [ind for ind, _ in scored[:POP_SIZE // 2]]  # elite selection
    offspring = []
    while len(offspring) < POP_SIZE // 2:
        parent = random.choice(population).copy()
        child = mutate(parent.copy())
        offspring.append(child)
    population += offspring

best = max(population, key=evaluate)
print("\n🎯 Best individual:", best, "Fitness:", evaluate(best))

# --- VHDL Code Generator ---
def generate_structural_vhdl(ind):
    gate = ind["gate"]
    in1, in2, in3 = ind["inputs"]

    vhdl = []
    vhdl.append("library IEEE;")
    vhdl.append("use IEEE.STD_LOGIC_1164.ALL;\n")
    vhdl.append("entity decoder_y_custom is")
    vhdl.append("    Port ( A, B, C : in STD_LOGIC;")
    vhdl.append("           Y : out STD_LOGIC );")
    vhdl.append("end decoder_y_custom;\n")

    vhdl.append("architecture Structural of decoder_y_custom is")
    signals = set()
    for i in [in1, in2, in3]:
        if i.startswith("n"):
            signals.add(i)
    if signals:
        vhdl.append(f"    signal {', '.join(signals)} : STD_LOGIC;")
    vhdl.append("    signal gate_out : STD_LOGIC;")
    vhdl.append("begin")

    # NOT operations
    for sig in signals:
        base = sig[1:]
        vhdl.append(f"    {sig} <= not {base};")

    # Gate logic
    if gate == "AND":
        vhdl.append(f"    gate_out <= {in1} and {in2} and {in3};")
    elif gate == "OR":
        vhdl.append(f"    gate_out <= {in1} or {in2} or {in3};")
    elif gate == "XOR":
        vhdl.append(f"    gate_out <= {in1} xor {in2} xor {in3};")
    elif gate == "NAND":
        vhdl.append(f"    gate_out <= not ({in1} and {in2} and {in3});")
    elif gate == "NOR":
        vhdl.append(f"    gate_out <= not ({in1} or {in2} or {in3});")
    else:
        vhdl.append("    gate_out <= '0';")

    vhdl.append("    Y <= gate_out;")
    vhdl.append("end Structural;")

    return "\n".join(vhdl)

# --- Output VHDL ---
vhdl_code = generate_structural_vhdl(best)

with open("decoder_y_custom.vhd", "w") as f:
    f.write(vhdl_code)

print("\n✅ VHDL code generated: decoder_y_custom.vhd")
print("\n📄 VHDL Preview:\n")
print(vhdl_code)
