import random

# --- GA Parameters ---
POP_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.2

# --- Inputs and Gates ---
INPUTS = ["A", "B", "C", "nA", "nB", "nC"]
GATES = ["AND", "OR", "XOR", "NAND", "NOR"]

# --- Get target 8-bit truth table from user ---
print("Enter target truth table for A,B,C (8 bits from 000 to 111):")
print("Example (Y3 = not A and B and C): 00100000")
user_bits = input("Target output (length 8): ").strip()
while len(user_bits) != 8 or any(c not in "01" for c in user_bits):
    user_bits = input("Invalid input. Please enter exactly 8 bits (0 or 1): ").strip()

target_outputs = [int(b) for b in user_bits]

def target_func(A, B, C):
    index = (A << 2) | (B << 1) | C
    return target_outputs[index]

# --- Individual Representation ---
def random_individual():
    return {
        "gate": random.choice(GATES),
        "inputs": random.sample(INPUTS, 3)
    }

# --- Fitness Evaluation ---
def evaluate(individual):
    score = 0
    for A in [0, 1]:
        for B in [0, 1]:
            for C in [0, 1]:
                vals = {
                    "A": A, "B": B, "C": C,
                    "nA": 1 - A, "nB": 1 - B, "nC": 1 - C
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
    return score  # Max 8

# --- Mutation ---
def mutate(ind):
    mutant = ind.copy()
    if random.random() < MUTATION_RATE:
        mutant["gate"] = random.choice(GATES)
    if random.random() < MUTATION_RATE:
        mutant["inputs"] = random.sample(INPUTS, 3)
    return mutant

# --- Evolution Loop ---
best = None
fitness = 0
attempts = 0

while fitness != 8 and attempts < 10:
    population = [random_individual() for _ in range(POP_SIZE)]
    for gen in range(GENERATIONS):
        population.sort(key=lambda ind: -evaluate(ind))
        best_fit = evaluate(population[0])
        if gen % 50 == 0 or best_fit == 8:
            print(f"Generation {gen}: Best fitness = {best_fit}")
        if best_fit == 8:
            break
        survivors = population[:POP_SIZE // 2]
        offspring = [mutate(random.choice(survivors)) for _ in range(POP_SIZE // 2)]
        population = survivors + offspring
    candidate = max(population, key=evaluate)
    fit = evaluate(candidate)
    print(f"\nAttempt {attempts + 1}: Fitness = {fit}")
    if fit == 8:
        best = candidate
        fitness = fit
        break
    attempts += 1

# --- Final Output ---
if best:
    print("\n🎯 Best individual found (fitness = 8 / 8):")
    print(f"{best['gate']}({best['inputs'][0]}, {best['inputs'][1]}, {best['inputs'][2]})")

    # --- Generate VHDL ---
    def generate_structural_vhdl(ind):
        gate = ind["gate"]
        in1, in2, in3 = ind["inputs"]
        inv_signals = {sig for sig in [in1, in2, in3] if sig.startswith("n")}

        lines = [
            "library IEEE;",
            "use IEEE.STD_LOGIC_1164.ALL;",
            "",
            "entity decoder_y_custom is",
            "  Port ( A, B, C : in STD_LOGIC;",
            "         Y       : out STD_LOGIC );",
            "end decoder_y_custom;",
            "",
            "architecture Structural of decoder_y_custom is"
        ]
        if inv_signals:
            lines.append(f"  signal {', '.join(sorted(inv_signals))} : STD_LOGIC;")
        lines.append("  signal gate_out : STD_LOGIC;")
        lines.append("begin")

        for sig in sorted(inv_signals):
            lines.append(f"  {sig} <= not {sig[1:]};")

        if gate == "AND":
            lines.append(f"  gate_out <= {in1} and {in2} and {in3};")
        elif gate == "OR":
            lines.append(f"  gate_out <= {in1} or {in2} or {in3};")
        elif gate == "XOR":
            lines.append(f"  gate_out <= {in1} xor {in2} xor {in3};")
        elif gate == "NAND":
            lines.append(f"  gate_out <= not ({in1} and {in2} and {in3});")
        elif gate == "NOR":
            lines.append(f"  gate_out <= not ({in1} or {in2} or {in3});")
        else:
            lines.append("  gate_out <= '0';")

        lines.append("  Y <= gate_out;")
        lines.append("end Structural;")

        return "\n".join(lines)

    vhdl_code = generate_structural_vhdl(best)
    with open("decoder_y_custom.vhd", "w") as f:
        f.write(vhdl_code)

    print("\n✅ VHDL code written to decoder_y_custom.vhd")

    print("\n🧪 Output Truth Table:")
    print(" A B C | Target Y | Output Y")
    for A in [0, 1]:
        for B in [0, 1]:
            for C in [0, 1]:
                vals = {"A": A, "B": B, "C": C, "nA": 1 - A, "nB": 1 - B, "nC": 1 - C}
                in1, in2, in3 = [vals[i] for i in best["inputs"]]
                if best["gate"] == "AND":
                    out = in1 & in2 & in3
                elif best["gate"] == "OR":
                    out = in1 | in2 | in3
                elif best["gate"] == "XOR":
                    out = in1 ^ in2 ^ in3
                elif best["gate"] == "NAND":
                    out = 1 - (in1 & in2 & in3)
                elif best["gate"] == "NOR":
                    out = 1 - (in1 | in2 | in3)
                else:
                    out = 0
                expected = target_func(A, B, C)
                print(f" {A} {B} {C} |     {expected}     |     {out}")

else:
    print("❌ Failed to evolve perfect decoder after 10 attempts.")
