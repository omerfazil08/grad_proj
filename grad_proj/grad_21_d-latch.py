import random

# --- Logic Gates ---
def AND(a, b): return a & b
def OR(a, b): return a | b
def XOR(a, b): return a ^ b
def NAND(a, b): return ~(a & b) & 1
def NOR(a, b): return ~(a | b) & 1
def XNOR(a, b): return ~(a ^ b) & 1
def NOT(a): return 1 - a

GATES = {
    'AND': AND,
    'OR': OR,
    'XOR': XOR,
    'NAND': NAND,
    'NOR': NOR,
    'XNOR': XNOR
}

# --- Target Truth Table for D-Latch ---
target_table = {
    (0, 0): (None, None),  # Hold
    (0, 1): (0, 1),
    (1, 0): (None, None),  # Hold
    (1, 1): (1, 0),
}

# --- Genome as a Gate Sequence ---
# Each gate is a tuple: (gate_type, input1_index, input2_index)
gate_types = [AND, OR, NOR, NOT]

# Each individual represents a logic circuit
def random_individual():
    # [Gate1, Gate2, ..., GateN]
    return [(
        random.choice(gate_types),
        random.randint(0, 3),  # Inputs: D=0, EN=1, PrevQ=2, PrevQn=3
        random.randint(0, 3)
    ) for _ in range(4)]  # 4 gates per individual

# --- Simulate Genome as Circuit ---
def simulate_individual(genome, D, EN, prev_Q=0):
    inputs = [D, EN, prev_Q, 1 - prev_Q]
    values = inputs[:]
    
    for gate, a, b in genome:
        if gate == NOT:
            result = gate(values[a])
        else:
            result = gate(values[a], values[b])
        values.append(result)
    
    # Last 2 values are assumed to be Q and Qn
    return values[-2], values[-1]

# --- Fitness Evaluation ---
def fitness(individual):
    score = 0
    for (D, EN), (target_Q, target_Qn) in target_table.items():
        prev_Q = 0  # Assume initial
        Q, Qn = simulate_individual(individual, D, EN, prev_Q)
        if target_Q is None: continue  # Ignore "hold" cases
        score += (Q == target_Q) + (Qn == target_Qn)
    return score

# --- Genetic Algorithm ---
def evolve(pop_size=100, generations=50):
    population = [random_individual() for _ in range(pop_size)]

    for gen in range(generations):
        scored = sorted(population, key=fitness, reverse=True)
        print(f"Gen {gen} - Best fitness: {fitness(scored[0])}/4")
        if fitness(scored[0]) == 4:
            print("Found perfect solution!")
            return scored[0]
        
        # Select top 20%, mutate and crossover
        survivors = scored[:pop_size // 5]
        children = []
        while len(children) < pop_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover(p1, p2)
            mutate(child)
            children.append(child)
        population = children
    return scored[0]

# --- Crossover and Mutation ---
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:]

def mutate(individual, rate=0.1):
    for i in range(len(individual)):
        if random.random() < rate:
            individual[i] = (
                random.choice(gate_types),
                random.randint(0, 3),
                random.randint(0, 3)
            )

# --- Run the Evolution ---
best = evolve()

# --- Show the Winning Logic ---
print("\nBest evolved logic:")
for gate in best:
    print(gate)
