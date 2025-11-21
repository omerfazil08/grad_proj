# main_colab_phase12.py
#
# Driver for evolution_colab_phase12.py
# Integrates Simplification (simplifier_phase12) at the end.

import matplotlib.pyplot as plt
from evolution_colab_phase12 import (
    ColabConfig,
    evolve_colab_phase12,
    print_results,
)
# --- NEW IMPORT ---
try:
    from simplifier_phase13 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    print("Warning: simplifier_phase12.py not found. Logic simplification disabled.")
    HAS_SIMPLIFIER = False


def get_user_target():
    """Interactive prompt (works both locally and in Colab cells)."""
    try:
        num_inputs = int(input("Enter number of inputs (2..8): ").strip())
        num_outputs = int(input("Enter number of outputs (1..4): ").strip())
    except ValueError:
        print("Invalid integer entered. Using default: 2 inputs, 1 output.")
        num_inputs, num_outputs = 2, 1

    print("\nInput rows order will be standard binary counting:")
    rows = []
    for i in range(2 ** num_inputs):
        tup = tuple((i >> b) & 1 for b in range(num_inputs - 1, -1, -1))
        rows.append(tup)
    for idx, r in enumerate(rows):
        print(f"{idx:2d}: {r}")
    print()

    targets = []
    for o in range(num_outputs):
        val_str = input(f"Enter output truth table #{o+1} "
                        f"({len(rows)} values 0/1 separated by spaces):\n→ ").strip()
        if not val_str:
            print("Empty input. Using zeros.")
            col = [0] * len(rows)
        else:
            vals = val_str.split()
            if len(vals) != len(rows):
                print(f"Warning: Expected {len(rows)} values, got {len(vals)}. Padding/Trimming.")
                vals = vals + ['0']*(len(rows)-len(vals))
                vals = vals[:len(rows)]
            col = [int(v) for v in vals]
        targets.append(col)

    return num_inputs, num_outputs, rows, targets


def run_simplification(best_individual, num_inputs, num_outputs):
    """Helper to run and print simplification results."""
    if not HAS_SIMPLIFIER:
        return

    print("\n🧠 Logic Simplification Analysis:")
    print("-" * 30)
    
    # Identify input names for display (A, B, C...)
    input_names = [chr(ord('A') + i) for i in range(num_inputs)]
    
    # Identify output gate names
    # (In phase12, outputs are always the last N gates)
    output_gate_names = [gate["output"] for gate in best_individual[-num_outputs:]]
    
    try:
        results = simplify_genome(best_individual, input_names, output_gate_names)
        for label, formula in results.items():
            print(f"{label}:  {formula}")
    except Exception as e:
        print(f"Simplification failed: {e}")
    print("-" * 30)


def run_ga_interactive():
    """Interactive GA run using the user-provided truth table."""
    n_in, n_out, inputs, targets = get_user_target()

    # --- DJ CONFIGURATION AREA (OPTIMIZED) ---
    cfg = ColabConfig(
        num_gates=24,
        pop_size=1200,
        generations=2000,
        elitism=15,
        tournament_k=8,
        base_mut=0.045,   # Optimized "Goldilocks" mutation
        min_mut=0.005,
        p_choose_primitive=0.60,

        log_every=50,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True,
        processes=None
    )

    best, score, max_score, history = evolve_colab_phase12(
        n_in, n_out, inputs, targets, cfg
    )

    print_results(best, score, max_score, n_out, inputs, targets)
    
    # --- SIMPLIFICATION STEP ---
    run_simplification(best, n_in, n_out)

    if history:
        plt.figure(figsize=(10, 6))
        plt.plot(history["gen"], history["best"], label="Best Fitness", color="blue")
        plt.plot(history["gen"], history["avg"], label="Avg Fitness", color="orange", alpha=0.7)
        plt.title("Evolutionary Gradient: Best vs Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Graph generated.")

def run_demo_full_adder():
    print("--- Running Full Adder Demo (Phase 12) ---")
    n_in = 4
    n_out = 3
    inputs = [( (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, (i >> 0) & 1 ) for i in range(16)]
    S1 =  [0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1]
    S0 =  [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0]
    Cout = [0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1]
    targets = [S1, S0, Cout]

    cfg = ColabConfig(
        num_gates=24,
        pop_size=1000,
        generations=1500,
        elitism=10,
        tournament_k=5,
        base_mut=0.04,
        min_mut=0.005,
        p_choose_primitive=0.60,
        log_every=50,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True,
        processes=None
    )

    best, score, max_score, history = evolve_colab_phase12(
        n_in, n_out, inputs, targets, cfg
    )

    print_results(best, score, max_score, n_out, inputs, targets)
    
    # --- SIMPLIFICATION STEP ---
    run_simplification(best, n_in, n_out)
    
    return best, score, max_score, history

def run_demo_xor():
    print("--- Running XOR Demo (Phase 12) ---")
    n_in = 2
    n_out = 1
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    targets = [[0, 1, 1, 0]]

    cfg = ColabConfig(
        num_gates=6,
        pop_size=200,
        generations=100,
        elitism=2,
        tournament_k=3,
        base_mut=0.10,
        min_mut=0.01,
        p_choose_primitive=0.80,
        log_every=10,
        record_history=True,
        seed=123,
        size_penalty_lambda=0.0,
        parallel=True
    )

    best, score, max_score, history = evolve_colab_phase12(
        n_in, n_out, inputs, targets, cfg
    )

    print_results(best, score, max_score, n_out, inputs, targets)
    
    # --- SIMPLIFICATION STEP ---
    run_simplification(best, n_in, n_out)
    
    return best, score, max_score, history


if __name__ == "__main__":
    print("=== Logic Evolution Phase 12 (Colab Optimized) ===")
    print("1) Interactive GA run")
    print("2) XOR demo")
    print("3) 4-input 3-output Full-Adder demo")
    
    choice = input("Select option (1/2/3): ").strip()

    if choice == "1":
        run_ga_interactive()
    elif choice == "3":
        run_demo_full_adder()
    else:
        run_demo_xor()