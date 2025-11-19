import random
import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, simplify_logic, Not, And, Or, Xor

# ---------------------------
# Configurable parameters
# ---------------------------
ENABLE_DIVERSITY = False      # Skip diversity calculation
SAVE_PLOTS_EVERY = 9999999    # No mid-run plotting
MAX_GATES = 16
GATES = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]

# ---------------------------
# Utility / Logic handling
# ---------------------------
def gate_eval(gate, values):
    if gate == "AND":  return int(all(values))
    if gate == "OR":   return int(any(values))
    if gate == "XOR":  return int(sum(values) % 2)
    if gate == "NAND": return int(not all(values))
    if gate == "NOR":  return int(not any(values))
    if gate == "XNOR": return int(not (sum(values) % 2))
    return 0

def random_gate(index, available):
    gate_type = random.choice(GATES)
    arity = random.randint(2, 3)
    inputs = random.sample(available, k=min(arity, len(available)))
    return {"name": f"g{index}", "gate": gate_type, "inputs": inputs}

def random_individual(num_gates=8, num_inputs=3):
    letters = ['A','B','C','D','E','F','G','H'][:num_inputs]
    available = [L for L in letters] + [f"n{L}" for L in letters]
    individual = []
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate["name"])
        individual.append(gate)
    return individual

def evaluate_network(individual, inp_dict):
    signals = inp_dict.copy()
    for name in list(inp_dict.keys()):
        if name.startswith("n"):
            continue
        signals[f"n{name}"] = 1 - inp_dict[name]

    for gate in individual:
        in_vals = [signals[inp] for inp in gate["inputs"]]
        signals[gate["name"]] = gate_eval(gate["gate"], in_vals)
    return signals

def evaluate_fitness(individual, num_inputs, num_outputs, inputs, targets):
    score = 0
    for i, row in enumerate(inputs):
        inp_dict = {chr(65 + j): row[j] for j in range(num_inputs)}
        signals = evaluate_network(individual, inp_dict)
        outputs = [signals.get(individual[-1]["name"], 0)] if num_outputs == 1 else \
                   [signals.get(individual[-k]["name"], 0) for k in range(1, num_outputs + 1)]
        score += sum(int(outputs[k] == targets[i][k]) for k in range(num_outputs))
    return score

def mutate(individual, gen, generations, num_inputs=3, base_mutation=0.25):
    mutant = []
    letters = ['A','B','C','D','E','F','G','H'][:num_inputs]
    available = [L for L in letters] + [f"n{L}" for L in letters]
    mutation_rate = base_mutation * (1 - gen / generations)
    for i, gate in enumerate(individual):
        if random.random() < mutation_rate:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate["name"])
        mutant.append(new_gate)
    return mutant

def crossover(p1, p2):
    cut = random.randint(1, len(p1) - 1)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

# ---------------------------
# Main Evolution Loop
# ---------------------------
def evolve_phase5_lite(inputs, targets, num_inputs, num_outputs, generations=1000, pop_size=800, num_gates=12):
    population = [random_individual(num_gates, num_inputs) for _ in range(pop_size)]
    max_score = len(inputs) * num_outputs
    fitness_hist = []

    for gen in range(generations):
        scores = [evaluate_fitness(ind, num_inputs, num_outputs, inputs, targets) for ind in population]
        best_idx = max(range(pop_size), key=lambda i: scores[i])
        best_score = scores[best_idx]
        fitness_hist.append(best_score)

        if gen % 20 == 0 or best_score == max_score:
            avg_fit = sum(scores) / pop_size
            print(f"Gen {gen:4d} | Best {best_score}/{max_score} | Avg {avg_fit:.2f} | Pop {pop_size}")
        if best_score == max_score:
            break

        next_gen = []
        while len(next_gen) < pop_size:
            parents = random.sample(population, 2)
            c1, c2 = crossover(parents[0], parents[1])
            next_gen.append(mutate(c1, gen, generations, num_inputs))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(c2, gen, generations, num_inputs))
        population = next_gen

    best = population[best_idx]
    plot_fitness(fitness_hist)
    draw_circuit(best)
    return best, fitness_hist, max_score

# ---------------------------
# Visualization
# ---------------------------
def plot_fitness(fitness_hist):
    plt.figure()
    plt.plot(fitness_hist, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitness_vs_gen.png")
    plt.close()

def draw_circuit(network):
    G = nx.DiGraph()
    for gate in network:
        for inp in gate["inputs"]:
            G.add_edge(inp, gate["name"])
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800)
    plt.title("Evolved Circuit Graph")
    plt.tight_layout()
    plt.savefig("circuit_graph.png")
    plt.close()
