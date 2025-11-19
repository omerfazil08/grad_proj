from evolution_phase5 import (
    evolve_phase5,
    evaluate_network,
    plot_evolution_stats,
    visualize_circuit,
    export_verilog
)

# ───────────────────────────────────────────────────────────────
# Input collection (same as before)
# ───────────────────────────────────────────────────────────────
def get_user_target():
    num_inputs = int(input("Enter number of inputs (2..8): "))
    num_outputs = int(input("Enter number of outputs (1..4): "))

    if num_inputs < 2 or num_inputs > 8:
        raise ValueError("Number of inputs must be 2–8.")
    if num_outputs < 1 or num_outputs > 4:
        raise ValueError("Number of outputs must be 1–4.")

    # Build input combinations
    from itertools import product
    inputs = list(product([0,1], repeat=num_inputs))
    print("\nInput rows order:")
    for r in inputs:
        print(r)

    targets = []
    for o in range(num_outputs):
        print(f"\nEnter output truth table #{o+1} ({len(inputs)} values):")
        vals = list(map(int, input("→ ").split()))
        if len(vals) != len(inputs):
            raise ValueError("Incorrect number of truth table entries.")
        targets.append(vals)
    return num_inputs, num_outputs, inputs, targets

# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    num_inputs, num_outputs, inputs, targets = get_user_target()

    best, fitness_hist, diversity_hist, max_score = evolve_phase5(
        num_inputs, num_outputs, inputs, targets,
        pop_size=800, generations=1000, num_gates=14
    )

    print("\n✅ Best network evolved successfully.")
    print("GATE LIST:")
    for g in best:
        print(f"{g['name']}: {g['gate']}({', '.join(g['inputs'])})")

    # Plot evolution
    plot_evolution_stats(fitness_hist, diversity_hist)

    # Visualize network
    visualize_circuit(best)

    # Export Verilog
    export_verilog(best, num_inputs, num_outputs)

    print("\n🎯 Phase 5 complete — stats, graph, and Verilog exported.")
