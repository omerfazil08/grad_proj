# main_colab_phase32.py
# High-Speed Bitwise Driver (Phase 3.2)

import matplotlib.pyplot as plt
import numpy as np
import sys
import vhdl_generator_1  # Ensure this is your valid VHDL generator
import evolution_colab_phase33  # <--- The new Locking Engine

# ... (Keep simplifier import if desired) ...
try:
    from simplifier_phase14 import simplify_genome
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False

# Helper
def get_user_target():
    # ... (Same input function as before) ...
    try:
        num_inputs = int(input("Enter number of inputs (2..12): ").strip())
        num_outputs = int(input("Enter number of outputs (1..10): ").strip())
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

def plot_search_gradient(history, filename="search_gradient.png"):
    if not history or 'mu' not in history: return
    generations = history['gen']
    best_fit = np.array(history['best'])
    mu = np.array(history['mu'])
    sigma = np.array(history['sigma'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(generations, best_fit, color='green', label='Best')
    ax1.plot(generations, mu, color='blue', label=r'Avg')
    ax1.fill_between(generations, mu-sigma, mu+sigma, color='blue', alpha=0.2)
    ax1.legend()
    ax2.plot(generations, sigma, color='orange')
    plt.savefig(filename)
    print("Plot saved.")

def run_5bit_adder_bitwise():
    n_in = 10; n_out = 6
    print(f"Generating 1024-row Truth Table...")
    inputs = []; targets = [[] for _ in range(n_out)]
    for i in range(1024):
        val_a = (i >> 5) & 0x1F
        val_b = i & 0x1F
        inp_bits = tuple((i >> b) & 1 for b in range(9, -1, -1))
        inputs.append(inp_bits)
        total = val_a + val_b
        for bit in range(n_out):
            targets[bit].append((total >> bit) & 1)

    # BitwiseConfig
    cfg = evolution_colab_phase33.BitwiseConfig(
        num_gates=60, 
        pop_size=3000,
        generations=6000, 
        elitism=500,
        tournament_k=10,
        base_mut=0.05, min_mut=0.005, 
        p_choose_primitive=0.60,
        log_every=10,
        record_history=True, 
        seed=42, 
        size_penalty_lambda=0.0, 
        parallel=True
    )
    
    print("🚀 Starting Phase 3.2 (Hardware Locking) Evolution...")
    
    best_ind_raw, best_bkd, hof, history = evolution_colab_phase33.evolve_bitwise(
        n_in, n_out, inputs, targets, cfg
    )
    
    print("Converting genome for export...")
    best_ind_str = evolution_colab_phase33.convert_to_string_format(best_ind_raw, n_in)
    
    print(f"Final Scores: {best_bkd}")
    
    # VHDL
    vhdl = vhdl_generator_1.generate_vhdl_code(best_ind_str, n_in, n_out)
    with open("evolved_5bit_adder.vhd", "w") as f:
        f.write(vhdl)
    print("✅ VHDL Exported.")
    
    plot_search_gradient(history)

if __name__ == "__main__":
    run_5bit_adder_bitwise()