{
    "gate": "AND",
    "inputs": ["nA", "B", "C"]
}
import random

# --- GA parametreleri ---
POP_SIZE = 30
GENERATIONS = 200
MUTATION_RATE = 0.2

# --- Mümkün giriş kaynakları ---
INPUTS = ["A", "B", "C", "nA", "nB", "nC"]
GATES = ["AND", "OR"]

# --- Hedef truth table (Y3) ---
def target_y3(A, B, C):
    return 1 if (A == 0 and B == 1 and C == 1) else 0

# --- Rastgele birey oluştur ---
def random_individual():
    return {
        "gate": random.choice(GATES),
        "inputs": random.sample(INPUTS, 3)
    }

# --- Bireyi değerlendir (fitness) ---
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
                if individual["gate"] == "AND":
                    out = in1 & in2 & in3
                else:
                    out = in1 | in2 | in3

                if out == target_y3(A, B, C):
                    score += 1
    return score  # 8 üzerinden

# --- Mutasyon ---
def mutate(ind):
    if random.random() < MUTATION_RATE:
        ind["gate"] = random.choice(GATES)
    if random.random() < MUTATION_RATE:
        ind["inputs"] = random.sample(INPUTS, 3)
    return ind

# --- GA döngüsü ---
population = [random_individual() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scored = [(ind, evaluate(ind)) for ind in population]
    scored.sort(key=lambda x: -x[1])
    population = [ind for ind, _ in scored[:POP_SIZE // 2]]  # elitizm
    offspring = []
    while len(offspring) < POP_SIZE // 2:
        parent = random.choice(population).copy()
        child = mutate(parent.copy())
        offspring.append(child)
    population += offspring

best = max(population, key=evaluate)
print("🎯 En iyi birey:", best, "Fitness:", evaluate(best))
def generate_structural_vhdl(ind):
    gate = ind["gate"]
    in1, in2, in3 = ind["inputs"]

    vhdl = []
    vhdl.append("library IEEE;")
    vhdl.append("use IEEE.STD_LOGIC_1164.ALL;\n")
    vhdl.append("entity decoder_y3 is")
    vhdl.append("    Port ( A, B, C : in STD_LOGIC;")
    vhdl.append("           Y3 : out STD_LOGIC );")
    vhdl.append("end decoder_y3;\n")

    vhdl.append("architecture Structural of decoder_y3 is")
    signals = set()
    for i in [in1, in2, in3]:
        if i.startswith("n"):
            signals.add(i)
    if signals:
        vhdl.append(f"    signal {', '.join(signals)} : STD_LOGIC;")
    vhdl.append("    signal gate_out : STD_LOGIC;")
    vhdl.append("begin")

    # NOT işlemleri
    for sig in signals:
        base = sig[1:]
        vhdl.append(f"    {sig} <= not {base};")

    # gate işlemi
    if gate == "AND":
        vhdl.append(f"    gate_out <= {in1} and {in2} and {in3};")
    else:
        vhdl.append(f"    gate_out <= {in1} or {in2} or {in3};")

    vhdl.append("    Y3 <= gate_out;")
    vhdl.append("end Structural;")

    return "\n".join(vhdl)

# VHDL üretimi
vhdl_code = generate_structural_vhdl(best)

# Dosyaya yaz
with open("decoder_y3_structural.vhd", "w") as f:
    f.write(vhdl_code)

# Konsola yaz
print("\n✅ VHDL kodu üretildi: decoder_y3_structural.vhd")
print("\n📄 Üretilen VHDL Kodu:\n")
print(vhdl_code)