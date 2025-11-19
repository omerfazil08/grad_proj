import random
import math
import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, simplify_logic, Not, And, Or, Xor, simplify

# ───────────────────────────────────────────────────────────────
# Logic Gate Definitions
# ───────────────────────────────────────────────────────────────
def AND(*args): return int(all(args))
def OR(*args): return int(any(args))
def XOR(a, b): return a ^ b
def NAND(*args): return int(not all(args))
def NOR(*args): return int(not any(args))
def XNOR(a, b): return int(not (a ^ b))

GATES = {
    'AND': AND,
    'OR': OR,
    'XOR': XOR,
    'NAND': NAND,
    'NOR': NOR,
    'XNOR': XNOR
}

# ───────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────
def random_gate(index, available_inputs):
    return {
        'name': f'g{index}',
        'gate': random.choice(list(GATES.keys())),
        'inputs': random.sample(available_inputs, min(2, len(available_inputs)))
    }

def random_individual(num_gates=8, num_inputs=4):
    letters = ['A','B','C','D','E','F','G','H'][:num_inputs]
    available = []
    for L in letters:
        available.extend([L, 'n' + L])
    individual = []
    for i in range(num_gates):
        gate = random_gate(i, available)
        available.append(gate['name'])
        individual.append(gate)
    return individual

def evaluate_network(individual, inputs_dict):
    signals = {**inputs_dict}
    for k in list(inputs_dict.keys()):
        if not k.startswith('n'):
            signals['n'+k] = 1 - signals[k]
    for gate in individual:
        in_vals = [signals[inp] for inp in gate['inputs']]
        signals[gate['name']] = GATES[gate['gate']](*in_vals)
    return signals

def evaluate_fitness(individual, num_inputs, num_outputs, inputs, targets):
    score = 0
    for idx, values in enumerate(inputs):
        inp_dict = {}
        letters = ['A','B','C','D','E','F','G','H'][:num_inputs]
        for i, v in enumerate(values):
            inp_dict[letters[i]] = v
        signals = evaluate_network(individual, inp_dict)
        for o in range(num_outputs):
            gate_name = individual[-num_outputs + o]['name']
            if signals[gate_name] == targets[o][idx]:
                score += 1
    return score

def population_diversity(pop):
    """Simple diversity metric: average pairwise Hamming distance between gate types."""
    if len(pop) < 2: return 0
    total = 0; count = 0
    for i in range(len(pop)):
        for j in range(i+1,len(pop)):
            gi = [g['gate'] for g in pop[i]]
            gj = [g['gate'] for g in pop[j]]
            d = sum(a!=b for a,b in zip(gi,gj))
            total += d; count += 1
    return total / count if count else 0

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2

def mutate(individual, gen, generations, num_inputs=4, base_mutation=0.25):
    mutant = []
    letters = ['A','B','C','D','E','F','G','H'][:num_inputs]
    available = []
    for L in letters:
        available.extend([L, 'n' + L])

    mutation_rate = base_mutation * (1 - gen / generations)
    for i, gate in enumerate(individual):
        if random.random() < mutation_rate:
            new_gate = random_gate(i, available.copy())
        else:
            new_gate = gate.copy()
        available.append(new_gate['name'])
        mutant.append(new_gate)
    return mutant

# ───────────────────────────────────────────────────────────────
# Evolution Process
# ───────────────────────────────────────────────────────────────
def evolve_phase5(num_inputs, num_outputs, inputs, targets,
                  pop_size=400, generations=600, num_gates=14):

    population = [random_individual(num_gates, num_inputs) for _ in range(pop_size)]

    max_score = len(inputs) * num_outputs
    fitness_history = []
    diversity_history = []

    for gen in range(generations):
        scores = [evaluate_fitness(ind, num_inputs, num_outputs, inputs, targets) for ind in population]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = population[best_idx]
        best_score = scores[best_idx]
        diversity = population_diversity(population)
        fitness_history.append(best_score)
        diversity_history.append(diversity)

        if gen % 20 == 0 or best_score == max_score:
            print(f"Gen {gen:4d} | Best {best_score}/{max_score} | Pop {pop_size} | Div {diversity:.1f}")

        if best_score == max_score:
            break

        next_gen = [best.copy()]  # elitism
        while len(next_gen) < pop_size:
            p1 = random.choice(population[:])
            p2 = random.choice(population[:])
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1, gen, generations, num_inputs))
            if len(next_gen) < pop_size:
                next_gen.append(mutate(c2, gen, generations, num_inputs))

        population = next_gen

    return best, fitness_history, diversity_history, max_score

# ───────────────────────────────────────────────────────────────
# Visualization Utilities
# ───────────────────────────────────────────────────────────────
def plot_evolution_stats(fitness_history, diversity_history, filename="evolution_stats.png"):
    plt.figure(figsize=(10,5))
    plt.plot(fitness_history, label="Best Fitness", linewidth=2)
    plt.plot(diversity_history, label="Diversity", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Evolution Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Saved evolution stats plot → {filename}")

def visualize_circuit(network, filename="circuit_graph.png"):
    G = nx.DiGraph()
    for gate in network:
        G.add_node(gate["name"], label=gate["gate"])
        for inp in gate["inputs"]:
            G.add_edge(inp, gate["name"])
    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(10,7))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color='lightblue', node_size=800, font_size=8, arrows=True)
    plt.title("Evolved Circuit Topology")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"🔧 Saved circuit graph → {filename}")

# ───────────────────────────────────────────────────────────────
# Verilog Exporter
# ───────────────────────────────────────────────────────────────
def export_verilog(network, num_inputs, num_outputs, filename="circuit.v"):
    letters = ['A','B','C','D','E','F','G','H'][:num_inputs]
    outs = [f"Y{o}" for o in range(num_outputs)]
    lines = []
    lines.append(f"module circuit(input {', '.join(letters)}, output {', '.join(outs)});")
    for gate in network:
        gtype = gate['gate']
        ins = ", ".join(gate['inputs'])
        out = gate['name']
        lines.append(f"  // {gtype}")
        if gtype == "AND":
            lines.append(f"  assign {out} = {ins.replace(',', ' &')};")
        elif gtype == "OR":
            lines.append(f"  assign {out} = {ins.replace(',', ' |')};")
        elif gtype == "XOR":
            lines.append(f"  assign {out} = {ins.replace(',', ' ^')};")
        elif gtype == "NAND":
            lines.append(f"  assign {out} = ~({ins.replace(',', ' &')});")
        elif gtype == "NOR":
            lines.append(f"  assign {out} = ~({ins.replace(',', ' |')});")
        elif gtype == "XNOR":
            lines.append(f"  assign {out} = ~({ins.replace(',', ' ^')});")
        else:
            lines.append(f"  // Unsupported gate type: {gtype}")
    lines.append("endmodule")
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"💾 Verilog file saved as {filename}")
