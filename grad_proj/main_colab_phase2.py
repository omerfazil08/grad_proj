# main_colab_phase2.py
# Driver for Incremental Evolution (Curriculum Learning)

import matplotlib.pyplot as plt
from evolution_colab_phase2 import (
    ColabConfig,
    evolve_colab_phase2,
    print_results_phase2,
)
try:
    from simplifier_phase12 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    print("Warning: simplifier_phase12.py not found.")
    HAS_SIMPLIFIER = False

def get_user_target():
    try:
        num_inputs = int(input("Enter number of inputs (2..8): ").strip())
        num_outputs = int(input("Enter number of outputs (1..4): ").strip())
    except ValueError:
        num_inputs, num_outputs = 2, 1
    
    rows = []
    for i in range(2 ** num_inputs):
        tup = tuple((i >> b) & 1 for b in range(num_inputs - 1, -1, -1))
        rows.append(tup)
        
    targets = []
    print(f"Input rows: {len(rows)}")
    for o in range(num_outputs):
        val_str = input(f"Enter output truth table #{o+1}:\n→ ").strip()
        vals = [int(v) for v in val_str.split()]
        targets.append(vals)
        
    return num_inputs, num_outputs, rows, targets

def run_demo_full_adder_incremental():
    print("--- Running Full Adder Demo (Phase 2: Incremental) ---")
    n_in = 3 # A, B, Cin
    n_out = 2 # Sum, Cout
    
    # 3-bit inputs
    inputs = [( (i>>2)&1, (i>>1)&1, (i>>0)&1 ) for i in range(8)]
    
    # Targets
    # Sum = A ^ B ^ C
    S = [0, 1, 1, 0, 1, 0, 0, 1]
    # Cout = Maj(A,B,C)
    C = [0, 0, 0, 1, 0, 1, 1, 1]
    
    targets = [S, C]

    cfg = ColabConfig(
        num_gates=16,
        pop_size=1000,
        generations=1000,
        elitism=15,
        tournament_k=6,
        base_mut=0.05,
        min_mut=0.005,
        p_choose_primitive=0.70,
        log_every=20,
        record_history=True,
        seed=42,
        size_penalty_lambda=0.0,
        parallel=True
    )

    best, best_bkd, hof = evolve_colab_phase2(
        n_in, n_out, inputs, targets, cfg
    )
    
    print_results_phase2(best, best_bkd, hof, inputs, targets)
    
    if HAS_SIMP:
        print("\n🧠 Simplification of Best Final Circuit:")
        input_names = ["A", "B", "Cin"]
        out_gates = [best[-(n_out-i)]["output"] for i in range(n_out)]
        res = simplify_genome(best, input_names, out_gates)
        for k, v in res.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    run_demo_full_adder_incremental()